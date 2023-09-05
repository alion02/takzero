use std::{array, fs::OpenOptions, io::Write, path::Path, sync::atomic::Ordering};

use arrayvec::ArrayVec;
use rand::{distributions::WeightedIndex, prelude::Distribution, Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use takzero::{
    network::Network,
    search::{
        env::Environment,
        node::{gumbel::gumbel_sequential_halving, Node},
    },
};
use tch::Device;

use crate::{
    new_opening,
    target::Replay,
    BetaNet,
    Env,
    Net,
    ReplayBuffer,
    MAXIMUM_REPLAY_BUFFER_SIZE,
    STEP,
};

const BATCH_SIZE: usize = 64;

const SAMPLED: usize = 32;
const SIMULATIONS: u32 = 512;

// This number should be large enough that there are also late-game positions.
const STEPS_BEFORE_CHECKING_NETWORK: usize = 500;

const WEIGHTED_RANDOM_PLIES: u16 = 30;

/// Populate the replay buffer with new state-action pairs from self-play.
pub fn run(
    device: Device,
    seed: u64,
    beta_net: &BetaNet,
    replay_buffer: &ReplayBuffer,
    replay_path: &Path,
    primary: bool,
) {
    log::debug!("started self-play thread, primary={primary}");

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let chacha_seed = rng.gen();

    let mut net = Net::new(device, None);
    let mut net_index = beta_net.0.load(Ordering::Relaxed);
    net.vs_mut().copy(&beta_net.1.read().unwrap()).unwrap();

    let mut envs: [_; BATCH_SIZE] = array::from_fn(|_| Env::default());
    let mut nodes: [_; BATCH_SIZE] = array::from_fn(|_| Node::default());
    let mut replays_batch: [_; BATCH_SIZE] = array::from_fn(|_| Vec::new());
    let mut actions: [_; BATCH_SIZE] = array::from_fn(|_| Vec::new());
    let mut trajectories: [_; BATCH_SIZE] = array::from_fn(|_| Vec::new());
    let mut rngs: [_; BATCH_SIZE] = array::from_fn(|i| {
        let mut rng = ChaCha8Rng::from_seed(chacha_seed);
        rng.set_stream(i as u64);
        rng
    });

    loop {
        self_play(
            &mut rng,
            &mut rngs,
            &net,
            &mut envs,
            &mut nodes,
            &mut replays_batch,
            &mut actions,
            &mut trajectories,
            replay_buffer,
        );

        // Truncate replay buffer if it gets too long.
        let mut lock = replay_buffer.write().unwrap();
        if lock.len() > MAXIMUM_REPLAY_BUFFER_SIZE {
            lock.truncate(MAXIMUM_REPLAY_BUFFER_SIZE);
        }
        drop(lock);

        //  Get the latest network
        log::info!("checking if there is a new model for self-play");
        let maybe_new_net_index = beta_net.0.load(Ordering::Relaxed);
        if maybe_new_net_index > net_index {
            net_index = maybe_new_net_index;
            net.vs_mut().copy(&beta_net.1.read().unwrap()).unwrap();
            log::info!("updating self-play model to beta{net_index}");

            // While doing this, also save the replay buffer
            if primary {
                let s: String = replay_buffer
                    .read()
                    .unwrap()
                    .iter()
                    .map(ToString::to_string)
                    .collect();
                let path = replay_path.join("replays.txt");
                std::thread::spawn(move || {
                    let mut file = OpenOptions::new()
                        .write(true)
                        .create(true)
                        .truncate(true)
                        .open(path)
                        .expect("replay file path should be valid and writable");
                    file.write_all(s.as_bytes()).unwrap();
                });
                log::debug!("saved replays to file");
            }
        }

        if cfg!(test) {
            break;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn self_play(
    rng: &mut impl Rng,
    rngs: &mut [ChaCha8Rng],
    agent: &Net,

    envs: &mut [Env],
    nodes: &mut [Node<Env>],
    replays_batch: &mut [Vec<Replay<Env>>],
    actions: &mut [Vec<<Env as Environment>::Action>],
    trajectories: &mut [Vec<usize>],

    replay_buffer: &ReplayBuffer,
) {
    envs.iter_mut()
        .zip(actions.iter_mut())
        .for_each(|(env, actions)| new_opening(env, actions, rng));
    nodes.iter_mut().for_each(|node| *node = Node::default());

    for _ in 0..STEPS_BEFORE_CHECKING_NETWORK {
        let mut top_actions = gumbel_sequential_halving(
            nodes,
            envs,
            agent,
            SAMPLED,
            SIMULATIONS,
            actions,
            trajectories,
            Some(rng),
        );
        // For openings, sample actions according to visits instead.
        envs.iter()
            .zip(rngs.iter_mut())
            .zip(nodes.iter_mut())
            .zip(&mut top_actions)
            .filter(|(((env, _), _), _)| env.steps() < WEIGHTED_RANDOM_PLIES)
            .for_each(|(((_, rng), node), top_action)| {
                let weighted_index =
                    WeightedIndex::new(node.children.iter().map(|(_, child)| child.visit_count))
                        .unwrap();
                *top_action = node.children[weighted_index.sample(rng)].0;
            });

        // Update replays.
        replays_batch
            .iter_mut()
            .zip(envs.iter())
            .zip(&top_actions)
            .for_each(|((replays, env), action)| {
                // Push start of fresh replay.
                replays.push(Replay {
                    env: env.clone(),
                    actions: ArrayVec::default(),
                });
                // Update existing replays.
                let from = replays.len().saturating_sub(STEP);
                for replay in &mut replays[from..] {
                    replay.actions.push(*action);
                }
            });

        // Take a step in environments and nodes.
        nodes
            .iter_mut()
            .zip(envs.iter_mut())
            .zip(top_actions)
            .for_each(|((node, env), action)| {
                node.descend(&action);
                env.step(action);
            });

        // Refresh finished environments and nodes.
        let mut lock = replay_buffer.write().unwrap();
        replays_batch
            .iter_mut()
            .zip(nodes.iter_mut())
            .zip(envs.iter_mut())
            .zip(actions.iter_mut())
            .filter_map(|(((replays, node), env), actions)| {
                env.terminal().map(|_| {
                    new_opening(env, actions, rng);
                    *node = Node::default();
                    replays.drain(..)
                })
            })
            .flatten()
            .for_each(|replay| lock.push_front(replay));
    }

    // Salvage replays from unfinished games.
    let mut lock = replay_buffer.write().unwrap();
    for replays in replays_batch {
        let len = replays.len().saturating_sub(STEP);
        replays
            .drain(..)
            .take(len)
            .for_each(|replay| lock.push_front(replay));
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::VecDeque,
        path::PathBuf,
        sync::{atomic::AtomicUsize, Arc, RwLock},
    };

    use rand::{Rng, SeedableRng};
    use takzero::network::Network;
    use tch::Device;

    use crate::{self_play::run, BetaNet, Net};

    // NOTE TO SELF:
    // Decrease constants above to actually see results before you die.
    #[test]
    fn self_play_works() {
        const SEED: u64 = 1234;

        let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);

        let mut net = Net::new(Device::Cpu, Some(rng.gen()));
        let beta_net: BetaNet = (AtomicUsize::new(0), RwLock::new(net.vs_mut()));

        let replay_buffer = Arc::new(RwLock::new(VecDeque::new()));

        run(
            Device::cuda_if_available(),
            rng.gen(),
            &beta_net,
            &replay_buffer,
            &PathBuf::default(),
            true,
        );

        for replay in &*replay_buffer.read().unwrap() {
            println!("{replay}");
        }
    }
}
