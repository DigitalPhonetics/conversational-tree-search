

use core::panic;
use std::{collections::{HashSet, HashMap}, hash::Hash};

use pest::{Parser, iterators::Pairs, iterators::Pair};
use super::variables::{VariableValue, FunctionEvaluator, VariableType};

#[derive(Parser)]
#[grammar = "parsers/answerParser.pest"]
struct AnswerParser;

pub struct AnswerTemplate {}

impl AnswerTemplate {
    pub fn get_variable_and_type(input: &str) -> (String, VariableType) {
        let pairs = AnswerParser::parse(Rule::rules, input).unwrap_or_else(|e| panic!("{}", e));

        let mut variable: Option<String> = None;
        let mut vartype: Option<VariableType> = None;
    
        for pair in pairs.into_iter() {
            match pair.as_rule() {
                Rule::identifier => { 
                    if variable == None {
                        variable = Some(pair.as_str().to_string()); 
                    } else {
                        panic!("Found more than 1 variable");
                    }
                }, // found variable
                Rule::vartype => {
                    if vartype == None {
                        vartype = Some(match pair.as_str() {
                            "TEXT" => VariableType::TEXT,
                            "NUMBER" => VariableType::NUMBER,
                            "LOCATION" => VariableType::LOCATION,
                            "BOOLEAN" => VariableType::BOOLEAN,
                            "TIMESPAN" => VariableType::TIMESPAN,
                            "TIMEPOINT" => VariableType::TIMEPOINT,
                            _ => {panic!("Unknown variable type {:?}", pair.as_str());},
                        });
                    } else {
                        panic!("Found more than 1 variable type");
                    }
                },
                _ => {}, // do nothing for other terminals
            }
        }
        return (variable.unwrap(), vartype.unwrap());
    }
}



#[cfg(test)]
mod tests {
    use crate::parsers::answer_parser::*;


    #[test]
    fn test_variable_finder() {
        let test_str = "{{ VAR1 = NUMBER }}";
        let (test_var, test_type) = AnswerTemplate::get_variable_and_type(test_str);
        assert_eq!(test_var, "VAR1");
        assert_eq!(test_type, VariableType::NUMBER);

        let test_str = "{{ VAR2 = TEXT }}";
        let (test_var, test_type) = AnswerTemplate::get_variable_and_type(test_str);
        assert_eq!(test_var, "VAR2");
        assert_eq!(test_type, VariableType::TEXT); 

        let test_str = "{{ var3 = BOOLEAN }}";
        let (test_var, test_type) = AnswerTemplate::get_variable_and_type(test_str);
        assert_eq!(test_var, "var3");
        assert_eq!(test_type, VariableType::BOOLEAN); 

        let test_str = "{{ Var4 = TIMESPAN }}";
        let (test_var, test_type) = AnswerTemplate::get_variable_and_type(test_str);
        assert_eq!(test_var, "Var4");
        assert_eq!(test_type, VariableType::TIMESPAN); 

        let test_str = "{{ Var5 = TIMEPOINT }}";
        let (test_var, test_type) = AnswerTemplate::get_variable_and_type(test_str);
        assert_eq!(test_var, "Var5");
        assert_eq!(test_type, VariableType::TIMEPOINT); 
    }

  
}