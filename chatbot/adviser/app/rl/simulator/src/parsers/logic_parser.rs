

use std::{collections::{HashSet, HashMap}, hash::Hash};

use pest::{Parser, iterators::Pairs, iterators::Pair};
use super::variables::{VariableValue, FunctionEvaluator};

#[derive(Parser)]
#[grammar = "parsers/logicParser.pest"]
struct LogicParser;

pub struct LogicTemplate {}

impl LogicTemplate {
    pub fn find_variables(input: &str) -> HashSet<String> {
        let pairs = LogicParser::parse(Rule::rules, input).unwrap_or_else(|e| panic!("{}", e));
        return LogicTemplate::recursive_find_variables(pairs);
    }

    fn recursive_find_variables(pairs: Pairs<Rule>) -> HashSet<String> {
        let mut variables: HashSet<String> = HashSet::new();
    
        for pair in pairs.into_iter() {
            match pair.as_rule() {
                Rule::identifier => { variables.insert(pair.as_str().to_string()); }, // found new variable
                Rule::conjunction | Rule::disjunction | Rule::multiplication | Rule::addition | Rule::condition  => { 
                    let res = LogicTemplate::recursive_find_variables(pair.into_inner());
                    variables.extend(res.iter().cloned());
                }, // recursion, could contain a variable
                Rule::function => {
                    // TABLE.FUNCTION(VAR1, ..., VARN) 
                    // -> skip variables that describe table and function names -> continue to fn_args (if any)
                    for fn_arg in pair.into_inner() {
                        match fn_arg.as_rule() {
                            Rule::identifier => {},
                            Rule::fn_args => { 
                                let res = LogicTemplate::recursive_find_variables(fn_arg.into_inner());
                                variables.extend(res.iter().cloned()); 
                            },
                            _ => { panic!("Expected function args in {:?}", fn_arg); }
                        }
                       
                    }
                }
                _ => {}, // do nothing for terminals
            }
        }
        return variables;
    }

    pub fn evaluate_template(input: &str, values: &HashMap<String, VariableValue>, functions: &HashMap<String, Box<dyn FunctionEvaluator>>) -> bool {
        let pairs = LogicParser::parse(Rule::rules, input).unwrap_or_else(|e| panic!("{}", e));
        return LogicTemplate::evaluate_disjunction(pairs.into_iter().next().unwrap(), values, functions); // skip EOI (last entry in first pairs)
    }
    
    fn eval_token(pair: Pair<Rule>, values: &HashMap<String, VariableValue>) -> VariableValue {
        match pair.as_rule() {
            Rule::identifier => values.get(pair.as_str()).unwrap().clone(),
            Rule::number => VariableValue::FloatVar(pair.as_str().parse::<f32>().unwrap()),
            Rule::negative_number => VariableValue::FloatVar(-pair.as_str().parse::<f32>().unwrap()),
            Rule::default => VariableValue::BoolVar(true),
            Rule::constant => VariableValue::StringVar(pair.into_inner().last().unwrap().as_str().to_string()),
            Rule::trueval => VariableValue::BoolVar(true),
            Rule::falseval => VariableValue::BoolVar(false),
            _ => panic!("Unknown leaf type to evaluate: {:?} with content {:?}", pair.as_rule(), pair.as_str()),
        }
    }

    fn perform_operation(lhs: f32, rhs: f32, op: &str) -> f32 {
        match op {
            "+" => lhs + rhs,
            "-" => lhs - rhs,
            "*" => lhs * rhs,
            "/" => lhs / rhs,
            _ => panic!("Unexpected operator rule: {:?} for lhs {:?} and rhs {:?}", op, lhs, rhs),
        }
    }

    fn eval_addition(pair: Pair<Rule>, values: &HashMap<String, VariableValue>, functions: &HashMap<String, Box<dyn FunctionEvaluator>>) -> VariableValue {
        assert_eq!(pair.as_rule(), Rule::addition);
        let mut inner : Vec<Pair<Rule>> = pair.into_inner().collect();
        if inner.len() == 1 {
            return LogicTemplate::evaluate_multiplication(inner.pop().unwrap(), values, functions);
        } else {
            // len >= 3 and we have integers here, otherwise panic since we would be trying to add e.g. strings
            let mut result = 0.0;
            let mut operation = String::from("+");
            for rule in inner.into_iter() {
                match rule.as_rule() {
                    Rule::multiplication => { 
                        if let VariableValue::FloatVar(rhs) = LogicTemplate::evaluate_multiplication(rule.clone(), values, functions) {
                            result = LogicTemplate::perform_operation(result, rhs, &operation);
                        } else {
                            panic!("Couldn't convert rhs to f32 {:?}: {:?}", rule.as_rule(), rule.as_str());
                        }
                    },
                    Rule::add | Rule::subtract => { operation = rule.as_str().to_string(); },
                    _ => {panic!("Unexpected addition rule {:?} with content {:?}", rule.as_rule(), rule.as_str()); },
                }
            }
            return VariableValue::FloatVar(result);
        }
    }

    fn evaluate_function(pair: Pair<Rule>, values: &HashMap<String, VariableValue>, functions: &HashMap<String, Box<dyn FunctionEvaluator>>) -> VariableValue {
        let mut fn_identifiers = Vec::new();
        let mut args = Vec::new();
        for fn_inner in pair.into_inner() {
            match fn_inner.as_rule() {
                Rule::identifier => { fn_identifiers.push(fn_inner.as_str()); },
                Rule::fn_args => {
                    for fn_arg in fn_inner.into_inner() {
                        match fn_arg.as_rule() {
                            Rule::addition => { args.push(LogicTemplate::eval_addition(fn_arg, values, functions)); },
                            _ => { panic!("Expected addition for function arg"); }
                        }
                    }
                },
                _ => { panic!("Expected identifier or function args"); }
            }
        }
        let fn_value = functions.get(&fn_identifiers.join(".")).unwrap().evaluate(&args);
        println!(" -> Evaluating function {:?} with args {:?}: {:?}", fn_identifiers.join("."), &args, &fn_value);
        return fn_value;
    }

    fn evaluate_multiplication(pair: Pair<Rule>, values: &HashMap<String, VariableValue>, functions: &HashMap<String, Box<dyn FunctionEvaluator>>) -> VariableValue {
        assert_eq!(pair.as_rule(), Rule::multiplication);
        let mut inner : Vec<Pair<Rule>> = pair.into_inner().collect();
        if inner.len() == 1 {
            match inner.last().unwrap().as_rule() {
                Rule::addition => { return LogicTemplate::eval_addition(inner.pop().unwrap(), values, functions); },
                Rule::function => { return LogicTemplate::evaluate_function(inner.pop().unwrap(), values, functions); },
                _ => { return LogicTemplate::eval_token(inner.pop().unwrap(), values); },
            }
        } else {
            // len >= 3 and we have integers here, otherwise panic since we would be trying to add e.g. strings
            let mut result = 1.0;
            let mut operation = String::from("*");
            for rule in inner.into_iter() {
                match rule.as_rule() {
                    Rule::addition => {
                        if let VariableValue::FloatVar(rhs) = LogicTemplate::eval_addition(rule.clone(), values, functions) {
                            result = LogicTemplate::perform_operation(result, rhs, &operation);
                        } else {
                            panic!("Couldn't convert rhs to f32 {:?}: {:?}", rule.as_rule(), rule.as_str());
                        }
                    },
                    Rule::multiply | Rule::divide => { operation = rule.as_str().to_string(); },
                    Rule::number | Rule::negative_number => { 
                        if let VariableValue::FloatVar(rhs) = LogicTemplate::eval_token(rule.clone(), values) {
                            result = LogicTemplate::perform_operation(result, rhs, &operation);
                        } else {
                            panic!("Couldn't convert rhs to f32 {:?}: {:?}", rule.as_rule(), rule.as_str());
                        }
                    },
                    _ => {panic!("Unexpected multiplication rule {:?} with content {:?}", rule.as_rule(), rule.as_str()); },
                }
            }
            return VariableValue::FloatVar(result);
        }
    }

    fn evaluate_disjunction(pair: Pair<Rule>, values: &HashMap<String, VariableValue>, functions: &HashMap<String, Box<dyn FunctionEvaluator>>) -> bool {
        assert_eq!(pair.as_rule(), Rule::disjunction);
        let inner : Vec<Pair<Rule>> = pair.into_inner().collect();
        let mut result = false;
        for rule in inner.into_iter() {
            match rule.as_rule() {
                Rule::conjunction => { 
                    let rhs = LogicTemplate::evaluate_conjunction(rule, values, functions);
                    result = result || rhs;
                    if result == true {
                        return result; // shorten calculation: evaluation will never change back to false
                    }
                },
                _ => {panic!("Unexpected addition rule {:?} with content {:?}", rule.as_rule(), rule.as_str()); },
            }
        }
        return result;
    }

    fn evaluate_conjunction(pair: Pair<Rule>, values: &HashMap<String, VariableValue>, functions: &HashMap<String, Box<dyn FunctionEvaluator>>) -> bool {
        assert_eq!(pair.as_rule(), Rule::conjunction);
        let inner : Vec<Pair<Rule>> = pair.into_inner().collect();
        let mut result = true;
        for rule in inner.into_iter() {
            match rule.as_rule() {
                Rule::condition => { 
                    let rhs = LogicTemplate::evaluate_condition(rule, values, functions);
                    result = result && rhs;
                    if result == false {
                        return result; // shorten calculation: evaluation will never change back to true
                    }
                },
                _ => {panic!("Unexpected addition rule {:?} with content {:?}", rule.as_rule(), rule.as_str()); },
            }
        }
        return result;
    }

    fn evaluate_condition (pair: Pair<Rule>, values: &HashMap<String, VariableValue>, functions: &HashMap<String, Box<dyn FunctionEvaluator>>) -> bool {
        assert_eq!(pair.as_rule(), Rule::condition);
        let inner : Vec<Pair<Rule>> = pair.into_inner().collect();
        if inner.len() == 1 {
            return LogicTemplate::evaluate_disjunction(inner.first().unwrap().to_owned(), values, functions);
        } else {
            // exactly 3 children
            assert_eq!(inner.len(), 3);
            let lhs = LogicTemplate::eval_addition(inner.get(0).unwrap().to_owned(), values, functions);
            let rhs = LogicTemplate::eval_addition(inner.get(2).unwrap().to_owned(), values, functions);
            let comperator = inner.get(1).unwrap();

            println!("Comparison: {:?} {:?} {:?}", lhs, comperator.as_str(), rhs);

            match comperator.as_rule() {
                Rule::eq => lhs == rhs,
                Rule::neq => lhs != rhs,
                Rule::le => lhs <= rhs,
                Rule::lt => lhs < rhs,
                Rule::ge => lhs >= rhs,
                Rule::gt => lhs > rhs,
                _ => {panic!("Unexpected comperator {:?} with text {:?}", comperator.as_rule(), comperator.as_str()); }
            }
        }
    }
  
}



#[cfg(test)]
mod tests {
    use crate::parsers::logic_parser::*;

    #[test]
    fn test_logic_variable_finder() {
        let variables = LogicTemplate::find_variables("{{ VAR1 != \"abc !\" AND (VAR2.VAR3() > 2 * 2 + -3) OR VAR2.VAR3(VAR4) > 5}}");
        assert_eq!(variables.len(), 2);
        let var1 = variables.get("VAR1");
        assert_ne!(var1, None);
        assert_eq!(var1.unwrap(), "VAR1");
        let var2 = variables.get("VAR4");
        assert_ne!(var2, None);
        assert_eq!(var2.unwrap(), "VAR4");
    }

    fn test_logic_evaluator_gt() {
        let functions: HashMap<String, Box<dyn FunctionEvaluator>> = HashMap::new();
        let test_str = "{{ 5 > 4 }}";
        let values: HashMap<String, VariableValue> = HashMap::new();
        let result = LogicTemplate::evaluate_template(test_str, &values, &functions);
        assert_eq!(result, true);

        let test_str = "{{ 4 > 5 }}";
        let values: HashMap<String, VariableValue> = HashMap::new();
        let result = LogicTemplate::evaluate_template(test_str, &values, &functions);
        assert_eq!(result, false);

        let test_str = "{{ 2*3 - 1 > 4}}";
        let values: HashMap<String, VariableValue> = HashMap::new();
        let result = LogicTemplate::evaluate_template(test_str, &values, &functions);
        assert_eq!(result, true);
    }

    #[test]
    fn test_logic_evaluator_lt() {
        let functions: HashMap<String, Box<dyn FunctionEvaluator>> = HashMap::new();
        let test_str = "{{ 5 < 4 }}";
        let values: HashMap<String, VariableValue> = HashMap::new();
        let result = LogicTemplate::evaluate_template(test_str, &values, &functions);
        assert_eq!(result, false);
    }

    #[test]
    fn test_logic_evaluator_string_comparion() {
        let functions: HashMap<String, Box<dyn FunctionEvaluator>> = HashMap::new();
        let test_str = "{{ \"TEST\" == \"TEST\" }}";
        let values: HashMap<String, VariableValue> = HashMap::new();
        let result = LogicTemplate::evaluate_template(test_str, &values, &functions);
        assert_eq!(result, true);

        let test_str = "{{ \"TEST\" == \"TEST2\" }}";
        let values: HashMap<String, VariableValue> = HashMap::new();
        let result = LogicTemplate::evaluate_template(test_str, &values, &functions);
        assert_eq!(result, false);

        let test_str = "{{ \"TEST\" != \"TEST2\" }}";
        let values: HashMap<String, VariableValue> = HashMap::new();
        let result = LogicTemplate::evaluate_template(test_str, &values, &functions);
        assert_eq!(result, true);

        let test_str = "{{ \"TEST\" != \"TEST\" }}";
        let values: HashMap<String, VariableValue> = HashMap::new();
        let result = LogicTemplate::evaluate_template(test_str, &values, &functions);
        assert_eq!(result, false);
    }

    #[test]
    fn test_logic_evaluator_variable_comparion() {
        let functions: HashMap<String, Box<dyn FunctionEvaluator>> = HashMap::new();
        let test_str = "{{ VAR1 == \"TEST\" }}";
        let mut values: HashMap<String, VariableValue> = HashMap::new();
        values.insert(String::from("VAR1"), VariableValue::StringVar(String::from("TEST")));
        let result = LogicTemplate::evaluate_template(test_str, &values, &functions);
        assert_eq!(result, true);

        let test_str = "{{ VAR1 == VAR2 }}";
        let mut values: HashMap<String, VariableValue> = HashMap::new();
        values.insert(String::from("VAR1"), VariableValue::StringVar(String::from("TEST")));
        values.insert(String::from("VAR2"), VariableValue::StringVar(String::from("TEST")));
        let result = LogicTemplate::evaluate_template(test_str, &values, &functions);
        assert_eq!(result, true);

        let test_str = "{{ VAR1 != VAR2 }}";
        let mut values: HashMap<String, VariableValue> = HashMap::new();
        values.insert(String::from("VAR1"), VariableValue::StringVar(String::from("TEST")));
        values.insert(String::from("VAR2"), VariableValue::StringVar(String::from("TEST2")));
        let result = LogicTemplate::evaluate_template(test_str, &values, &functions);
        assert_eq!(result, true);

        let test_str = "{{ VAR1 == VAR2 }}";
        let mut values: HashMap<String, VariableValue> = HashMap::new();
        values.insert(String::from("VAR1"), VariableValue::StringVar(String::from("TEST")));
        values.insert(String::from("VAR2"), VariableValue::StringVar(String::from("TEST2")));
        let result = LogicTemplate::evaluate_template(test_str, &values, &functions);
        assert_eq!(result, false);

        let test_str = "{{ VAR1 > VAR2 }}";
        let mut values: HashMap<String, VariableValue> = HashMap::new();
        values.insert(String::from("VAR1"), VariableValue::FloatVar(1.));
        values.insert(String::from("VAR2"), VariableValue::FloatVar(-1.));
        let result = LogicTemplate::evaluate_template(test_str, &values, &functions);
        assert_eq!(result, true);

        let test_str = "{{ 10 > VAR2 }}";
        let mut values: HashMap<String, VariableValue> = HashMap::new();
        values.insert(String::from("VAR2"), VariableValue::FloatVar(-1.));
        let result = LogicTemplate::evaluate_template(test_str, &values, &functions);
        assert_eq!(result, true);
    }

    #[test]
    fn test_logic_evaluator_eq() {
        let functions: HashMap<String, Box<dyn FunctionEvaluator>> = HashMap::new();
        let test_str = "{{ (2*3)/2 + 2*2 == ((5-1)+1+2)}}";
        let values: HashMap<String, VariableValue> = HashMap::new();
        let result = LogicTemplate::evaluate_template(test_str, &values, &functions);
        assert_eq!(result, true);

        let test_str = "{{ (2*3)/2 + 2*3 == ((5-1)+1+2)}}";
        let values: HashMap<String, VariableValue> = HashMap::new();
        let result = LogicTemplate::evaluate_template(test_str, &values, &functions);
        assert_eq!(result, false);
    }

    #[test]
    fn test_logic_evaluator_neq() {
        let functions: HashMap<String, Box<dyn FunctionEvaluator>> = HashMap::new();
        let test_str = "{{ (2*3)/2 + 2*2 != ((5-1)+1+2)}}";
        let values: HashMap<String, VariableValue> = HashMap::new();
        let result = LogicTemplate::evaluate_template(test_str, &values, &functions);
        assert_eq!(result, false);

        let test_str = "{{ (2*3)/2 + 2*3 != ((5-1)+1+2)}}";
        let values: HashMap<String, VariableValue> = HashMap::new();
        let result = LogicTemplate::evaluate_template(test_str, &values, &functions);
        assert_eq!(result, true);
    }

    #[test]
    fn test_logic_evaluator_default() {
        let functions: HashMap<String, Box<dyn FunctionEvaluator>> = HashMap::new();
        let test_str = "{{ DEFAULT == TRUE }}";
        let values: HashMap<String, VariableValue> = HashMap::new();
        let result = LogicTemplate::evaluate_template(test_str, &values, &functions);
        assert_eq!(result, true);

        let test_str = "{{ DEFAULT != TRUE }}";
        let values: HashMap<String, VariableValue> = HashMap::new();
        let result = LogicTemplate::evaluate_template(test_str, &values, &functions);
        assert_eq!(result, false);
    }

    #[test]
    fn test_logic_evaluator_logic() {
        let functions: HashMap<String, Box<dyn FunctionEvaluator>> = HashMap::new();
        let test_str = "{{ DEFAULT == TRUE AND 1 == 1 }}";
        let values: HashMap<String, VariableValue> = HashMap::new();
        let result = LogicTemplate::evaluate_template(test_str, &values, &functions);
        assert_eq!(result, true);

        let test_str = "{{ DEFAULT == FALSE AND 1 == 1 }}";
        let values: HashMap<String, VariableValue> = HashMap::new();
        let result = LogicTemplate::evaluate_template(test_str, &values, &functions);
        assert_eq!(result, false);

        let test_str = "{{ 1 > 2 OR 1 < 2 }}";
        let values: HashMap<String, VariableValue> = HashMap::new();
        let result = LogicTemplate::evaluate_template(test_str, &values, &functions);
        assert_eq!(result, true);

        let test_str = "{{ 1 > 2 AND 1 < 2 }}";
        let values: HashMap<String, VariableValue> = HashMap::new();
        let result = LogicTemplate::evaluate_template(test_str, &values, &functions);
        assert_eq!(result, false);
    }

    #[test]
    fn test_logic_evaluator_conjunction_disjunction() {
        let functions: HashMap<String, Box<dyn FunctionEvaluator>> = HashMap::new();
        let test_str = "{{ (DEFAULT == TRUE AND 1 == 1) OR 1 == 2 }}";
        let values: HashMap<String, VariableValue> = HashMap::new();
        let result = LogicTemplate::evaluate_template(test_str, &values, &functions);
        assert_eq!(result, true);

        let test_str = "{{ (DEFAULT == TRUE OR 1 == 1) AND 1 == 1 }}";
        let values: HashMap<String, VariableValue> = HashMap::new();
        let result = LogicTemplate::evaluate_template(test_str, &values, &functions);
        assert_eq!(result, true);
    }

    struct TestFunction {}
    impl FunctionEvaluator for TestFunction {
        fn evaluate(&self, args: &Vec<VariableValue>) -> VariableValue {
            let mut sum = 0.0;
            for arg in args {
                if let VariableValue::FloatVar(val) = arg {
                    sum += val;
                } else {
                    panic!("Expected float value in test function");
                }
            }
            return VariableValue::FloatVar(sum);
        }
    }

    #[test]
    fn test_logic_function_evaluation() {
        let mut functions: HashMap<String, Box<dyn FunctionEvaluator>> = HashMap::new();
        functions.insert(String::from("TABLE1.FUNCTION1"), Box::new(TestFunction{}));
        let test_str = "{{ TABLE1.FUNCTION1(1,2,3,4) == 10 }}";
        let values: HashMap<String, VariableValue> = HashMap::new();
        let result = LogicTemplate::evaluate_template(test_str, &values, &functions);
        assert_eq!(result, true);

    }

}