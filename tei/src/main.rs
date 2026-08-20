use std::{
    num::NonZeroUsize,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::TryRecvError,
        Arc,
    },
    time::{Duration, Instant},
};

#[cfg(feature = "playtak-policy")]
use clap::Parser;
use fast_tak::takparse::{Color, Move};
use protocol::{GoOption, Id, Input, Output, ParseInputError, Position, ValueType};
#[cfg(feature = "playtak-policy")]
use rand::Rng;
use takzero::{
    network::{
        net6_simhash::{Env, Net, HALF_KOMI, N},
        Network,
    },
    search::{
        agent::Agent,
        env::Environment,
        eval::Eval,
        node::{policy::softmax, Node},
    },
};
use thiserror::Error;

mod protocol;

const MAX_ERRORS_IN_A_ROW: usize = 5;
const DURATION_BETWEEN_INFO_PRINTS: Duration = Duration::from_millis(300);
const DURATION_BEFORE_CHECKING_INPUT: Duration = Duration::from_secs(1);
const BATCH_SIZE: usize = 128;
const BETA: f32 = 0.0;

#[cfg(feature = "playtak-policy")]
#[derive(Debug, Parser)]
struct PlaytakPolicyArgs {
    /// Wait this many milliseconds before returning each move.
    #[arg(long, alias = "move-time-ms", default_value_t = 0)]
    fixed_move_time_ms: u64,
    /// Probability mass to retain when sampling a move.
    #[arg(long, default_value_t = 1.0)]
    top_p: f32,
    /// Temperature used when sampling a move.
    #[arg(long, default_value_t = 1.0)]
    temperature: f32,
    /// Linear top-p decrease per played ply.
    #[arg(long, default_value_t = 0.0)]
    top_p_decay_per_ply: f32,
    /// Linear temperature decrease per played ply.
    #[arg(long, default_value_t = 0.0)]
    temperature_decay_per_ply: f32,
    /// Model path.
    #[arg(long)]
    model: String,
}

#[allow(clippy::too_many_lines)] // FIXME
#[allow(clippy::cognitive_complexity)] // FIXME
fn main() {
    env_logger::init();
    #[cfg(feature = "playtak-policy")]
    let playtak_policy_args = PlaytakPolicyArgs::parse();
    let mut line = String::new();
    let stdin = std::io::stdin();

    // Wait for first `tei` message.
    let Ok(Input::Tei) = get_input(&stdin, &mut line) else {
        log::error!("first message received should be `tei`");
        return;
    };

    // Print name and author.
    println!("{}", Output::Id(Id::Name("TakZero")));
    println!("{}", Output::Id(Id::Author("Viliam Vadocz (0x57696c6c)")));

    // Describe engine options.
    #[cfg(not(feature = "playtak-policy"))]
    println!("{}", Output::Option {
        name: "model",
        value_type: ValueType::String,
        default: Some("./path/to/model.ot"),
        min: None,
        max: None,
        variables: &[]
    });
    println!("{}", Output::Option {
        name: "HalfKomi",
        value_type: ValueType::Combo,
        default: Some("4"),
        min: None,
        max: None,
        variables: &["4"]
    });
    println!("{}", Output::Option {
        name: "MultiPV",
        value_type: ValueType::Spin,
        default: Some("5"),
        min: Some("1"),
        max: Some("2048"),
        variables: &[]
    });
    #[cfg(not(feature = "playtak-policy"))]
    println!("{}", Output::Option {
        name: "PolicyOnly",
        value_type: ValueType::Check,
        default: Some("false"),
        min: None,
        max: None,
        variables: &[]
    });

    println!("{}", Output::Ok);

    // Configure engine options.
    let mut model_path =
        cfg!(feature = "playtak-policy").then_some(playtak_policy_args.model.clone());
    let mut num_multi_pv = 5;
    let mut policy_only = cfg!(feature = "playtak-policy");
    loop {
        match get_input(&stdin, &mut line) {
            Ok(Input::IsReady) => break,
            Ok(Input::Option { name, value }) => match name.as_ref() {
                "model" => model_path = Some(value),
                "HalfKomi" => {
                    let Ok(half_komi) = value.parse::<i8>() else {
                        log::error!("could not parse half komi");
                        return;
                    };
                    if half_komi != HALF_KOMI {
                        log::error!(
                            "half komi of {half_komi} was requested, but only {HALF_KOMI} is \
                             supported"
                        );
                        return;
                    }
                }
                "MultiPV" => {
                    let Ok(x) = value.parse::<usize>() else {
                        log::error!("could not parse multi pv");
                        return;
                    };
                    num_multi_pv = x;
                }
                #[cfg(not(feature = "playtak-policy"))]
                "PolicyOnly" => {
                    let Ok(x) = value.parse::<bool>() else {
                        log::error!("could not parse policy only");
                        return;
                    };
                    policy_only = x;
                }
                _ => log::warn!("unknown option: {name}"),
            },
            Ok(_) => log::warn!("only expecting `isready` or `option` messages"),
            Err(err) => log::error!("{err}"),
        }
    }

    // Validate configuration.
    let Some(model_path) = model_path else {
        log::error!("model path must be set");
        return;
    };

    // Load engine / model.
    let device = tch::Device::cuda_if_available();
    if !device.is_cuda() {
        log::warn!("CUDA is not available, running on CPU");
    }
    let net = match Net::load_partial(model_path, device) {
        Ok(net) => net,
        Err(err) => {
            log::error!("failed to load model: {err}");
            return;
        }
    };

    // Start thread to parse user input and send it over.
    let (tx, rx) = std::sync::mpsc::channel();
    let should_stop = Arc::new(AtomicBool::new(false));
    let should_stop_2 = should_stop.clone();
    let input_thread = std::thread::spawn(move || {
        let should_stop = should_stop_2;
        let mut errors_in_a_row = 0;
        while !should_stop.load(Ordering::Relaxed) {
            match get_input(&stdin, &mut line) {
                Ok(x) => tx.send(x).expect("Main thread should still be alive."),
                Err(err) => {
                    log::error!("{err}");
                    errors_in_a_row += 1;
                    if errors_in_a_row >= MAX_ERRORS_IN_A_ROW {
                        log::error!("there were {MAX_ERRORS_IN_A_ROW} errors in a row");
                        should_stop.store(true, Ordering::Relaxed);
                    }
                    continue;
                }
            }
            errors_in_a_row = 0;
        }
    });

    // Ready!
    println!("{}", Output::ReadyOk);

    let mut node = Node::default();
    let mut env = Env::default();
    node.simulate_simple(&net, env.clone(), 0.0);
    let mut go_status = GoStatus::Stopped;
    let mut go_options = Vec::new();

    let mut nodes = None;
    let mut move_time = None;
    let mut my_time = None;
    let mut my_inc = None;
    let mut visits_at_start = 0;
    let mut start = Instant::now();
    let mut last_info = Instant::now();
    let mut sent_info = false;

    let mut last_position: Position = Position::StartPos;
    let mut last_moves: Vec<Move> = vec![];

    let print_info = |start: Instant, visits_at_start: u32, node: &mut Node<_>| {
        let elapsed = start.elapsed();
        node.sort_actions_best_to_worst();
        println!("{}", Output::Info {
            time: elapsed,
            nodes_since_start: (node.visit_count - visits_at_start) as _,
            nodes: node.visit_count as _,
            score: node.evaluation,
            principal_variation: node.principal_variation().collect(),
            multi_pv: None,
            cp: None,
        });
        for (multi_pv, (action, child)) in node.children.iter().rev().take(num_multi_pv).enumerate()
        {
            println!("{}", Output::Info {
                time: elapsed,
                nodes_since_start: 0, // TODO?
                nodes: child.visit_count as _,
                score: child.evaluation.negate(),
                principal_variation: std::iter::once(*action)
                    .chain(child.principal_variation())
                    .collect(),
                multi_pv: Some(NonZeroUsize::new(1 + multi_pv).unwrap()),
                cp: None,
            });
        }
    };

    'main_loop: while !should_stop.load(Ordering::Relaxed) {
        // Process user input
        let last_checked_input = Instant::now();
        match if matches!(go_status, GoStatus::Stopped) {
            rx.recv().map_err(|_| TryRecvError::Disconnected)
        } else {
            rx.try_recv()
        } {
            Ok(Input::IsReady) => println!("{}", Output::ReadyOk),
            Ok(Input::NewGame { size }) => {
                if size != N {
                    log::error!("the engine is compiled only for size {N}");
                    break 'main_loop;
                }
                node = Node::default();
                env = Env::default();
            }
            Ok(Input::Position { position, moves }) => {
                if position == last_position && moves.starts_with(&last_moves) {
                    // Tree re-use!
                    let new_moves = &moves[last_moves.len()..];
                    for my_move in new_moves {
                        node.descend(my_move);
                        if let Err(err) = env.play(*my_move) {
                            log::error!("could not play move {my_move}: {err}");
                            break;
                        }
                    }
                } else {
                    // Restart
                    node = Node::default();
                    env = match &position {
                        Position::StartPos => Env::default(),
                        Position::Tps(tps) => tps.clone().into(),
                    };
                    for &my_move in &moves {
                        if let Err(err) = env.play(my_move) {
                            log::error!("could not play move {my_move}: {err}");
                            break;
                        }
                    }
                }
                last_position = position;
                last_moves = moves;
            }
            Ok(Input::Stop) => {
                go_status = GoStatus::Stopping;
            }
            Ok(Input::Go(options)) => {
                if policy_only {
                    // Time controls in `go` are disregarded in policy-only mode.
                    // Run the policy network once on the current position and
                    // report the top moves sorted by policy score.
                    let start = Instant::now();
                    let mut actions = Vec::new();
                    env.populate_actions(&mut actions);
                    let (policy, ..) = net
                        .policy_value_uncertainty(
                            std::slice::from_ref(&env),
                            std::slice::from_ref(&actions),
                        )
                        .next()
                        .expect("agent should return exactly one prediction");
                    #[cfg(feature = "playtak-policy")]
                    let raw_policy: Vec<(Move, f32)> = policy
                        .iter()
                        .map(|(mv, value)| (*mv, (*value).into_inner()))
                        .collect();
                    // Normalize the raw network policy for display and ranking.
                    let probabilities = softmax(policy.clone().into_iter().map(|(_, p)| p));
                    let mut moves: Vec<(Move, i32)> = policy
                        .into_iter()
                        .zip(probabilities)
                        // cp is repurposed to carry the normalized policy
                        // value scaled 0-100.
                        .map(|((mv, _), p)| (mv, (p.into_inner() * 100.0).round() as i32))
                        .collect();
                    // Sort by policy score, highest first.
                    moves.sort_by(|(_, a), (_, b)| b.cmp(a));
                    let elapsed = start.elapsed();
                    if moves.is_empty() {
                        log::error!("no legal moves in policy-only mode");
                    } else {
                        for (multi_pv, (mv, policy_value)) in
                            moves.iter().take(num_multi_pv).enumerate()
                        {
                            println!("{}", Output::Info {
                                time: elapsed,
                                nodes_since_start: 1,
                                nodes: 1,
                                score: Eval::default(),
                                principal_variation: vec![*mv],
                                multi_pv: Some(NonZeroUsize::new(1 + multi_pv).unwrap()),
                                cp: Some(*policy_value),
                            });
                        }
                        #[cfg(feature = "playtak-policy")]
                        let best_move =
                            sample_policy_move(&raw_policy, &playtak_policy_args, last_moves.len());
                        #[cfg(not(feature = "playtak-policy"))]
                        let best_move = moves[0].0;

                        #[cfg(feature = "playtak-policy")]
                        if let Some(remaining) =
                            Duration::from_millis(playtak_policy_args.fixed_move_time_ms)
                                .checked_sub(start.elapsed())
                        {
                            std::thread::sleep(remaining);
                        }
                        println!("{}", Output::BestMove(best_move));
                    }
                    go_status = GoStatus::Stopped;
                } else {
                    go_options.clear();
                    go_options.extend(options);
                    go_status = GoStatus::Starting;
                }
            }
            Ok(Input::Option { .. }) => log::warn!("it's too late to specify options"),
            Ok(Input::Tei) => log::warn!("tei does not make sense here"),
            Ok(Input::Quit) | Err(TryRecvError::Disconnected) => break 'main_loop,
            Err(TryRecvError::Empty) => {}
        }

        if matches!(go_status, GoStatus::Starting) {
            for option in go_options.drain(..) {
                match option {
                    GoOption::Nodes(amount) => nodes = Some(amount),
                    GoOption::MoveTime(duration) => move_time = Some(duration),
                    GoOption::WhiteTime(duration) if env.to_move == Color::White => {
                        my_time = Some(duration);
                    }
                    GoOption::BlackTime(duration) if env.to_move == Color::Black => {
                        my_time = Some(duration);
                    }
                    GoOption::WhiteIncrement(duration) if env.to_move == Color::White => {
                        my_inc = Some(duration);
                    }
                    GoOption::BlackIncrement(duration) if env.to_move == Color::Black => {
                        my_inc = Some(duration);
                    }
                    GoOption::Infinite => nodes = Some(usize::MAX), // HACK
                    _ => log::warn!("ignored `go` option {option:?}"),
                }
            }
            if nodes.is_none() && move_time.is_none() && (my_time.is_none() || my_inc.is_none()) {
                log::warn!("no understood stopping condition given");
            }
            // Very basic time management.
            if let (None, Some(my_time), Some(my_inc)) = (move_time, my_time, my_inc) {
                move_time = Some(my_time / 10 + 3 * my_inc / 4);
            }
            visits_at_start = node.visit_count;
            sent_info = false;
            last_info = Instant::now();
            start = Instant::now();
            go_status = GoStatus::Going;
        }

        if matches!(go_status, GoStatus::Going) {
            loop {
                node.simulate_batch(&net, &env, BETA, BATCH_SIZE);
                let visits = (node.visit_count - visits_at_start) as usize;
                let elapsed = start.elapsed();

                let done = nodes.is_some_and(|amount| visits >= amount)
                    || move_time.is_some_and(|duration| elapsed >= duration);

                if last_info.elapsed() >= DURATION_BETWEEN_INFO_PRINTS {
                    print_info(start, visits_at_start, &mut node);
                    sent_info = true;
                    last_info = Instant::now();
                }
                if done {
                    go_status = GoStatus::Stopping;
                    break;
                }
                // Go check for `stop`.
                if last_checked_input.elapsed() >= DURATION_BEFORE_CHECKING_INPUT {
                    continue 'main_loop;
                }
            }
        }

        if matches!(go_status, GoStatus::Stopping) {
            if !sent_info {
                print_info(start, visits_at_start, &mut node);
            }
            println!("{}", Output::BestMove(node.select_best_action()));
            nodes = None;
            move_time = None;
            my_time = None;
            my_inc = None;
            go_status = GoStatus::Stopped;
        }
    }

    should_stop.store(true, Ordering::Relaxed);
    input_thread
        .join()
        .expect("Input thread should shut down gracefully.");
}

enum GoStatus {
    Stopped,
    Starting,
    Going,
    Stopping,
}

#[cfg(feature = "playtak-policy")]
fn sample_policy_move(policy: &[(Move, f32)], args: &PlaytakPolicyArgs, ply: usize) -> Move {
    let ply = ply as f32;
    let temperature = (args.temperature - args.temperature_decay_per_ply * ply).max(f32::EPSILON);
    let top_p = (args.top_p - args.top_p_decay_per_ply * ply).clamp(f32::EPSILON, 1.0);

    let max = policy
        .iter()
        .map(|(_, logit)| *logit)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut probabilities: Vec<(Move, f32)> = policy
        .iter()
        .map(|(mv, logit)| (*mv, ((*logit - max) / temperature).exp()))
        .collect();
    let sum: f32 = probabilities
        .iter()
        .map(|(_, probability)| *probability)
        .sum();
    probabilities
        .iter_mut()
        .for_each(|(_, probability)| *probability /= sum);
    probabilities.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

    let mut retained = Vec::new();
    let mut cumulative = 0.0;
    for (mv, probability) in probabilities {
        retained.push((mv, probability));
        cumulative += probability;
        if cumulative >= top_p {
            break;
        }
    }

    let retained_mass: f32 = retained.iter().map(|(_, probability)| *probability).sum();
    let sample = rand::rng().random_range(0.0..retained_mass);
    let mut cumulative = 0.0;
    retained
        .into_iter()
        .find(|(_, probability)| {
            cumulative += *probability;
            sample < cumulative
        })
        .map_or_else(|| policy[0].0, |(mv, _)| mv)
}

#[derive(Debug, Error)]
enum GetInputError {
    #[error("reading from stdin failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("parse error: {0}")]
    Parse(#[from] ParseInputError),
}

fn get_input(stdin: &std::io::Stdin, line: &mut String) -> Result<Input, GetInputError> {
    loop {
        line.clear();
        stdin.read_line(line)?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        return Ok(trimmed.parse()?);
    }
}
