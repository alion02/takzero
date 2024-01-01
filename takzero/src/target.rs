use std::{collections::VecDeque, fmt, num::ParseFloatError, str::FromStr};

use fast_tak::{
    takparse::{ParseMoveError, ParseTpsError, Tps},
    Game,
    Reserves,
    Symmetry,
};
use rand::prelude::*;
use thiserror::Error;

use crate::search::{env::Environment, node::Node};

#[derive(Debug, PartialEq)]
pub struct Target<E: Environment> {
    pub env: E,                          // s_t
    pub policy: Box<[(E::Action, f32)]>, // \pi'(s_t)
    pub value: f32,                      // discounted N-step value
    pub ube: f32,                        // sum of RND + discounted N-step UBE
}

pub trait Augment {
    #[must_use]
    fn augment(&self, rng: &mut impl Rng) -> Self;
}

impl<const N: usize, const HALF_KOMI: i8> Augment for Target<Game<N, HALF_KOMI>>
where
    Reserves<N>: Default,
{
    fn augment(&self, rng: &mut impl Rng) -> Self {
        let index = rng.gen_range(0..8);
        Self {
            env: self.env.symmetries().into_iter().nth(index).unwrap(),
            value: self.value,
            ube: self.ube,
            policy: self
                .policy
                .iter()
                .map(|(mov, p)| (Symmetry::<N>::symmetries(mov)[index], *p))
                .collect(),
        }
    }
}

impl<const N: usize, const HALF_KOMI: i8> fmt::Display for Target<Game<N, HALF_KOMI>>
where
    Reserves<N>: Default,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let tps: Tps = self.env.clone().into();
        let value = self.value;
        let ube = self.ube;
        let policy = self
            .policy
            .iter()
            .map(|(mov, p)| format!("{mov}:{p}"))
            .collect::<Vec<_>>()
            .join(",");

        writeln!(f, "{tps};{value};{ube};{policy}")
    }
}

#[derive(Error, Debug)]
pub enum ParseTargetError {
    #[error("missing TPS")]
    MissingTps,
    #[error("missing value")]
    MissingValue,
    #[error("missing UBE")]
    MissingUbe,
    #[error("missing policy")]
    MissingPolicy,
    #[error("policy format is wrong")]
    WrongPolicyFormat,
    #[error("{0}")]
    Tps(#[from] ParseTpsError),
    #[error("{0}")]
    Action(#[from] ParseMoveError),
    #[error("{0}")]
    Float(#[from] ParseFloatError),
}

impl<const N: usize, const HALF_KOMI: i8> FromStr for Target<Game<N, HALF_KOMI>>
where
    Reserves<N>: Default,
{
    type Err = ParseTargetError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        //{tps};{value};{ube};{policy}
        let mut iter = s.trim().split(';');
        let tps: Tps = iter.next().ok_or(ParseTargetError::MissingTps)?.parse()?;
        let value = iter.next().ok_or(ParseTargetError::MissingValue)?.parse()?;
        let ube = iter.next().ok_or(ParseTargetError::MissingUbe)?.parse()?;
        let policy = iter
            .next()
            .ok_or(ParseTargetError::MissingPolicy)?
            .split(',')
            .map(|s| {
                s.split_once(':')
                    .ok_or(ParseTargetError::WrongPolicyFormat)
                    .and_then(|(a, p)| Ok((a.parse()?, p.parse()?)))
            })
            .collect::<Result<_, _>>()?;

        Ok(Self {
            env: tps.into(),
            policy,
            value,
            ube,
        })
    }
}

#[must_use]
pub fn policy_target_from_proportional_visits<E: Environment>(
    node: &Node<E>,
) -> Box<[(E::Action, f32)]> {
    node.children
        .iter()
        .map(|(action, child)| {
            (
                action.clone(),
                child.visit_count as f32 / node.visit_count as f32,
            )
        })
        .collect()
}

pub struct Replay<E: Environment> {
    pub env: E,
    pub actions: VecDeque<E::Action>,
}

impl<E: Environment> Replay<E> {
    /// Start a new replay from an initial position.
    pub fn new(env: E) -> Self {
        Self {
            env,
            actions: VecDeque::new(),
        }
    }

    /// Add an action to the replay.
    pub fn push(&mut self, action: E::Action) {
        self.actions.push_back(action);
    }

    pub fn len(&self) -> usize {
        self.actions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }

    /// Advance the replay state by `steps`.
    ///
    /// # Panics
    ///
    /// Panics if there are fewer actions in the replay than requested steps.
    pub fn advance(&mut self, steps: usize) {
        for _ in 0..steps {
            self.env.step(self.actions.pop_front().unwrap());
        }
    }
}

impl<const N: usize, const HALF_KOMI: i8> fmt::Display for Replay<Game<N, HALF_KOMI>>
where
    Reserves<N>: Default,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", Tps::from(self.env.clone()),)?;
        for action in &self.actions {
            write!(f, " {action}")?;
        }
        writeln!(f)
    }
}

#[derive(Error, Debug)]
pub enum ParseReplayError {
    #[error("missing TPS")]
    MissingTps,
    #[error("{0}")]
    Tps(#[from] ParseTpsError),
    #[error("{0}")]
    Action(#[from] ParseMoveError),
}

impl<const N: usize, const HALF_KOMI: i8> FromStr for Replay<Game<N, HALF_KOMI>>
where
    Reserves<N>: Default,
{
    type Err = ParseReplayError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut iter = s.split_whitespace();
        let tps: Tps = iter.next().ok_or(ParseReplayError::MissingTps)?.parse()?;
        let env = tps.into();
        Ok(Self {
            env,
            actions: iter.map(str::parse).collect::<Result<_, _>>()?,
        })
    }
}

#[cfg(test)]
mod tests {
    use fast_tak::Game;
    use rand::{seq::IteratorRandom, Rng, SeedableRng};

    use crate::{search::env::Environment, target::Target};

    #[test]
    fn target_consistency() {
        const SEED: u64 = 123;
        let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);
        let mut env: Game<5, 4> = Game::default();
        let mut actions = Vec::new();
        while env.terminal().is_none() {
            env.populate_actions(&mut actions);
            let replay = Target {
                env: {
                    let mut c = env.clone();
                    c.reversible_plies = 0;
                    c
                },
                policy: actions.iter().map(|a| (*a, rng.gen())).collect(),
                value: rng.gen(),
                ube: rng.gen(),
            };
            let string = replay.to_string();
            println!("{string}");

            let recovered: Target<_> = string.parse().unwrap();
            let string_again = recovered.to_string();

            assert_eq!(replay, recovered);
            assert_eq!(string, string_again);

            env.step(actions.drain(..).choose(&mut rng).unwrap());
        }
    }
}
