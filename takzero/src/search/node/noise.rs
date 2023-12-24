use ordered_float::NotNan;
use rand::Rng;
use rand_distr::{Dirichlet, Distribution};

use super::Node;
use crate::search::env::Environment;

impl<E: Environment> Node<E> {
    #[allow(clippy::missing_panics_doc)]
    pub fn apply_dirichlet(&mut self, rng: &mut impl Rng, alpha: f32, ratio: f32) {
        assert!(
            self.visit_count > 0,
            "cannot apply dirichlet noise without initialized policy"
        );
        let dirichlet = Dirichlet::new(&vec![alpha; self.children.len()]).unwrap();
        let samples = dirichlet.sample(rng);

        self.children
            .iter_mut()
            .zip(samples)
            .for_each(|((_, child), noise)| {
                child.probability = child.probability * (1.0 - ratio) + noise * ratio;
                child.logit = NotNan::new(child.probability.ln())
                    .expect("Logit from probability should not be NaN");
            });
    }
}
