//! Fine-tune a TakZero policy towards human-like play.
//!
//! Reads high-skill human games (tab-separated `game_id<TAB>notation`, one game
//! per line, where notation is the tak-server move list such as
//! `P A1,P F1,M D3 D2 1,...`), replays them with `fast_tak`, and collects
//! (position, human move) pairs. The base model is loaded from `--model` and
//! fine-tuned with cross-entropy on the human moves, training only the late
//! residual blocks and the policy head (everything else is frozen), which
//! keeps memory and gradient computation low and allows larger batch sizes.

use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::PathBuf,
};

use clap::Parser;
use fast_tak::{
    takparse::{Direction, Move, MoveKind, Piece, Square},
    Game,
    Symmetry,
};
use rand::{prelude::SliceRandom, rngs::StdRng, SeedableRng};
use takzero::network::{
    net6_simhash::{Env, Net, CORE_RES_BLOCKS, N},
    repr::{game_to_tensor, move_index, output_size},
    Network,
};
use tch::{
    nn::{Adam, OptimizerConfig},
    Device,
    Kind,
    Tensor,
};

/// A training sample: a position and the move a human played from it.
struct Sample {
    game: Env,
    mv: Move,
}

#[derive(Parser, Debug)]
struct Args {
    /// TSV file(s) with `game_id<TAB>notation` lines. Can be passed multiple times.
    #[arg(long)]
    data: Vec<PathBuf>,
    /// Base model to fine-tune from.
    #[arg(long)]
    model: PathBuf,
    /// Where to save the fine-tuned model (`.ot`).
    #[arg(long)]
    out: PathBuf,
    /// Number of residual blocks to freeze. The rest plus the policy head train.
    #[arg(long, default_value_t = 13)]
    frozen_blocks: usize,
    /// Batch size.
    #[arg(long, default_value_t = 256)]
    batch_size: usize,
    /// Number of training steps.
    #[arg(long, default_value_t = 2000)]
    steps: usize,
    /// Learning rate.
    #[arg(long, default_value_t = 1e-3)]
    lr: f64,
    /// Augment each sample with the 8 board symmetries (8x data).
    #[arg(long)]
    augment: bool,
    /// Log every N steps.
    #[arg(long, default_value_t = 10)]
    log_every: usize,
    /// Overwrite the output model every N steps.
    #[arg(long, default_value_t = 250)]
    save_every: usize,
    /// CUDA device id.
    #[arg(long, default_value_t = 0)]
    device: usize,
    /// Random seed.
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Instead of training, evaluate the given model on all samples
    /// (reporting average cross-entropy loss and top-1 accuracy).
    #[arg(long)]
    eval: bool,
}

/// Parse a tak-server square (e.g. `D3`) into a `takparse::Square`.
fn parse_square(s: &str) -> Option<Square> {
    let mut chars = s.chars();
    let column = chars.next()? as u8;
    let row = chars.next()? as u8;
    if chars.next().is_some() {
        return None;
    }
    let column = column.wrapping_sub(b'A');
    let row = row.wrapping_sub(b'1');
    if column < N as u8 && row < N as u8 {
        Some(Square::new(column, row))
    } else {
        None
    }
}

/// The direction and number of steps from `from` to `to`.
/// Moves in Tak are along a single axis.
fn direction(from: Square, to: Square) -> Option<(Direction, u32)> {
    let (from_column, from_row) = (from.column(), from.row());
    let (to_column, to_row) = (to.column(), to.row());
    if to_row > from_row {
        Some((Direction::Up, u32::from(to_row - from_row)))
    } else if to_row < from_row {
        Some((Direction::Down, u32::from(from_row - to_row)))
    } else if to_column > from_column {
        Some((Direction::Right, u32::from(to_column - from_column)))
    } else if to_column < from_column {
        Some((Direction::Left, u32::from(from_column - to_column)))
    } else {
        None
    }
}

/// Parse a single tak-server move token (e.g. `P A1`, `P E4 C`, `P D5 W`,
/// `M D3 D2 1`, `M B4 B2 1 2`) into a `takparse::Move`.
///
/// tak-server drops `vals` on successive squares after the source, so the
/// takparse PTN form `<count><square><direction><drops>` matches directly.
fn parse_token(token: &str) -> Option<Move> {
    let parts: Vec<&str> = token.split_whitespace().collect();
    if parts.is_empty() {
        return None;
    }
    match parts[0] {
        "P" => {
            let square = parse_square(parts.get(1)?)?;
            let piece = match parts.get(2).copied() {
                Some("C") => Piece::Cap,
                Some("W") => Piece::Wall,
                _ => Piece::Flat,
            };
            Some(Move::new(square, MoveKind::Place(piece)))
        }
        "M" => {
            let from = parse_square(parts.get(1)?)?;
            let to = parse_square(parts.get(2)?)?;
            let drops: Vec<u32> = parts[3..]
                .iter()
                .map(|d| d.parse().ok())
                .collect::<Option<Vec<_>>>()?;
            if drops.is_empty() {
                return None;
            }
            let (dir, _steps) = direction(from, to)?;
            let total = drops.iter().sum::<u32>();
            let drops: String = drops.iter().map(u32::to_string).collect();
            format!("{total}{from}{dir}{drops}").parse().ok()
        }
        _ => None,
    }
}

/// Parse a tak-server game notation into a list of moves.
fn parse_notation(notation: &str) -> Option<Vec<Move>> {
    notation
        .split(',')
        .map(str::trim)
        .map(parse_token)
        .collect()
}

/// Replay the games and collect (position, human move) pairs.
fn load_samples(paths: &[PathBuf], augment: bool) -> Vec<Sample> {
    let mut samples = Vec::new();
    let mut games = 0usize;
    let mut failed_moves = 0usize;
    for path in paths {
        let file = File::open(path).unwrap_or_else(|e| panic!("cannot open {}: {e}", path.display()));
        for line in BufReader::new(file).lines() {
            let line = line.unwrap();
            let mut fields = line.split('\t');
            let _game_id = fields.next();
            let Some(notation) = fields.next() else { continue };
            let Some(moves) = parse_notation(notation) else { continue };
            if moves.is_empty() {
                continue;
            }
            let mut game: Env = Game::default();
            for mv in &moves {
                let before = game.clone();
                if game.play(*mv).is_ok() {
                    samples.push(Sample { game: before, mv: *mv });
                } else {
                    failed_moves += 1;
                    break;
                }
            }
            games += 1;
        }
    }
    log::info!("replayed {games} games ({failed_moves} moves could not be replayed)");
    if augment {
        let mut augmented = Vec::with_capacity(samples.len() * 8);
        for sample in &samples {
            let games = sample.game.symmetries();
            let moves = Symmetry::<N>::symmetries(&sample.mv);
            for (game, mv) in games.into_iter().zip(moves) {
                augmented.push(Sample { game, mv });
            }
        }
        samples = augmented;
    }
    samples
}

/// Evaluate the model on all samples in eval mode and log the average
/// cross-entropy loss and top-1 accuracy.
fn evaluate(
    net: &Net,
    samples: &[Sample],
    batch_size: usize,
    device: Device,
    frozen_blocks: usize,
) {
    let mut total_loss = 0.0f64;
    let mut total_correct = 0usize;
    let mut num_batches = 0usize;
    for batch in samples.chunks(batch_size) {
        num_batches += 1;
        let inputs = Tensor::cat(
            &batch
                .iter()
                .map(|s| game_to_tensor(&s.game, device))
                .collect::<Vec<_>>(),
            0,
        );
        let targets: Vec<i64> = batch
            .iter()
            .map(|s| move_index::<N>(&s.mv) as i64)
            .collect();
        let target = Tensor::from_slice(&targets).to(device);

        let logits = net
            .forward_policy_finetune(&inputs, false, frozen_blocks)
            .view([-1, output_size::<N>() as i64]);
        total_loss += f64::try_from(logits.cross_entropy_for_logits(&target)).unwrap();
        total_correct += i64::try_from(
            logits
                .argmax(1, false)
                .eq_tensor(&target)
                .sum(Kind::Int64),
        )
        .unwrap() as usize;
    }
    log::info!(
        "eval on {} samples: avg_loss = {:.4}, top-1 acc = {:.4}",
        samples.len(),
        total_loss / num_batches as f64,
        total_correct as f64 / samples.len() as f64
    );
}

/// Whether a variable (by name) should be trained.
/// Only the policy head and the last `CORE_RES_BLOCKS - frozen_blocks`
/// residual blocks are trainable.
fn is_trainable(name: &str, frozen_blocks: usize) -> bool {
    // Batch norm running statistics and counters are never trainable.
    if name.contains(".batch_norm.running_") || name.contains(".batch_norm.num_batches_tracked") {
        return false;
    }
    if let Some(rest) = name.strip_prefix("core.res_block_") {
        rest.split('.')
            .next()
            .and_then(|index| index.parse::<usize>().ok())
            .is_some_and(|index| index >= frozen_blocks)
    } else {
        name.starts_with("policy.")
    }
}

fn main() {
    env_logger::init();
    let args = Args::parse();
    let device = Device::Cuda(args.device);
    assert!(
        args.frozen_blocks <= CORE_RES_BLOCKS,
        "--frozen-blocks must be at most {CORE_RES_BLOCKS}"
    );
    assert!(args.batch_size > 0, "--batch-size must be positive");

    let samples = load_samples(&args.data, args.augment);
    log::info!(
        "loaded {} samples from {} file(s)",
        samples.len(),
        args.data.len()
    );
    assert!(!samples.is_empty(), "no training samples were loaded");

    let net = Net::load_partial(&args.model, device)
        .unwrap_or_else(|e| panic!("cannot load base model {}: {e}", args.model.display()));

    if args.eval {
        evaluate(&net, &samples, args.batch_size, device, args.frozen_blocks);
        return;
    }
    assert!(args.steps > 0, "--steps must be positive");
    let mut net = net;

    // Freeze everything except the policy head and the late residual blocks.
    let (mut frozen, mut trainable) = (0usize, 0usize);
    for (name, var) in net.vs().variables() {
        let train = is_trainable(&name, args.frozen_blocks);
        let _ = var.set_requires_grad(train);
        if train {
            trainable += 1;
        } else {
            frozen += 1;
        }
        log::debug!("{name} trainable={train}");
    }
    log::info!("frozen {frozen} variables, training {trainable} variables");

    let mut opt = Adam::default().build(net.vs_mut(), args.lr).unwrap();

    let mut rng = StdRng::seed_from_u64(args.seed);
    let mut order: Vec<usize> = (0..samples.len()).collect();
    order.shuffle(&mut rng);
    let mut cursor = 0usize;

    // Make sure the output directory exists.
    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }

    let mut cumulative_loss = 0.0f64;
    let mut cumulative_acc = 0.0f64;
    for step in 1..=args.steps {
        if cursor + args.batch_size > samples.len() {
            order.shuffle(&mut rng);
            cursor = 0;
        }
        let batch = &order[cursor..cursor + args.batch_size];
        cursor += args.batch_size;

        let inputs = Tensor::cat(
            &batch
                .iter()
                .map(|&i| game_to_tensor(&samples[i].game, device))
                .collect::<Vec<_>>(),
            0,
        );
        let targets: Vec<i64> = batch
            .iter()
            .map(|&i| move_index::<N>(&samples[i].mv) as i64)
            .collect();
        let target = Tensor::from_slice(&targets).to(device);

        let logits = net
            .forward_policy_finetune(&inputs, true, args.frozen_blocks)
            .view([-1, output_size::<N>() as i64]);
        let loss = logits.cross_entropy_for_logits(&target);
        opt.backward_step(&loss);

        let loss_value = f64::try_from(loss).unwrap();
        let correct = logits
            .argmax(1, false)
            .eq_tensor(&target)
            .sum(Kind::Int64);
        let accuracy = f64::try_from(correct).unwrap() / args.batch_size as f64;
        cumulative_loss += loss_value;
        cumulative_acc += accuracy;

        if step % args.log_every == 0 {
            #[rustfmt::skip]
            log::info!(
                "step {step}/{}  loss = {loss_value:.4}  acc = {accuracy:.4}  \
                 avg_loss = {:.4}  avg_acc = {:.4}",
                args.steps,
                cumulative_loss / args.log_every as f64,
                cumulative_acc / args.log_every as f64
            );
            cumulative_loss = 0.0;
            cumulative_acc = 0.0;
        }

        if step % args.save_every == 0 {
            net.save(&args.out).unwrap();
            log::info!("saved model to {}", args.out.display());
        }
    }

    net.save(&args.out).unwrap();
    log::info!("saved final model to {}", args.out.display());
}
