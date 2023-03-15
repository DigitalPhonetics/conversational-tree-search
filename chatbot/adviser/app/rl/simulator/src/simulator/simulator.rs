
use std::cell::Cell;

use crate::{data::types::Dataset, parsers::{logic_parser::LogicTemplate, answer_parser::AnswerTemplate}};

pub struct Simulator {
    dataset: Dataset,
    reward_normalization: f32,

    num_guided_dialogs: Cell<i64>,
    num_freeform_dialogs: Cell<i64>,

}




impl Simulator {
    fn new(ds: Dataset, stop_action: bool, max_steps: Option<u8>, user_patience: u8, normalize_rewards: bool, stop_when_reaching_goal: bool, dialog_faq_ratio: f32, 
            log_to_file: Option<String>, env_id: i8, return_obs: bool) -> Self {
        
        let _max_steps = match max_steps {
            Some(steps) => steps,
            None => 2 * ds.max_tree_depth,
        };


        
                
        panic!("NOT IMPLEMENTED");
    }
}
