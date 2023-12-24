use std::fmt;

use ordered_float::NotNan;

use super::{
    super::{env::Environment, eval::Eval},
    Node,
};

// TODO: Improve this
impl<E: Environment> fmt::Display for Node<E>
where
    E::Action: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut action_info = self.action_info(0.0);
        action_info.sort_by_key(|a| a.improved_policy);
        writeln!(
            f,
            "[root]   c:{: >8} v:{:+.4} e:{:+.4}",
            self.visit_count, self.variance, self.evaluation
        )?;
        for a in action_info {
            writeln!(f, "{a}")?;
        }
        Ok(())
    }
}

impl<E: Environment> Node<E> {
    #[must_use]
    pub fn action_info(&self, beta: f32) -> Vec<ActionInfo<E::Action>> {
        self.improved_policy(beta)
            .zip(self.children.iter())
            .map(|(improved_policy, (action, child))| ActionInfo {
                action: action.clone(),
                visit_count: child.visit_count,
                logit: child.logit,
                improved_policy,
                eval: child.evaluation,
                variance: child.variance,
            })
            .collect()
    }
}

pub struct ActionInfo<A> {
    action: A,
    visit_count: u32,
    logit: NotNan<f32>,
    improved_policy: NotNan<f32>,
    variance: NotNan<f32>,
    eval: Eval,
}

impl<A: fmt::Display> fmt::Display for ActionInfo<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{: >8} c:{: >8} l:{:+.4} i:{:+.4} v:{:.4} e:{:+.4?}",
            self.action.to_string(),
            self.visit_count,
            f32::from(self.logit),
            f32::from(self.improved_policy),
            f32::from(self.variance),
            self.eval
        )
    }
}
