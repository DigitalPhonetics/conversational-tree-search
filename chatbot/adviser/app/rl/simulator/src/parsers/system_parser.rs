

use std::{collections::{HashSet, HashMap}, hash::Hash};

use pest::{Parser, iterators::Pairs, iterators::Pair};
use super::variables::{VariableValue, FunctionEvaluator};

#[derive(Parser)]
#[grammar = "parsers/systemParser.pest"]
struct SystemParser;

pub struct SystemTemplate {}

impl SystemTemplate {
    pub fn find_variables(input: &str) -> HashSet<String> {
        let pairs = SystemParser::parse(Rule::rules, input).unwrap_or_else(|e| panic!("{}", e));
        return SystemTemplate::recursive_find_variables(pairs);
    }

    fn recursive_find_variables(pairs: Pairs<Rule>) -> HashSet<String> {
        let mut variables: HashSet<String> = HashSet::new();
    
        for pair in pairs.into_iter() {
            match pair.as_rule() {
                Rule::identifier => { variables.insert(pair.as_str().to_string()); }, // found new variable
                Rule::multiplication | Rule::addition  => { 
                    let res = SystemTemplate::recursive_find_variables(pair.into_inner());
                    variables.extend(res.iter().cloned());
                }, // recursion, could contain a variable
                Rule::function => {
                    // TABLE.FUNCTION(VAR1, ..., VARN) 
                    // -> skip variables that describe table and function names -> continue to fn_args (if any)
                    for fn_arg in pair.into_inner() {
                        match fn_arg.as_rule() {
                            Rule::identifier => {},
                            Rule::fn_args => { 
                                let res = SystemTemplate::recursive_find_variables(fn_arg.into_inner());
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

    pub fn evaluate_template(input: &str, values: &HashMap<String, VariableValue>, functions: &HashMap<String, Box<dyn FunctionEvaluator>>) -> String {
        let pairs = SystemParser::parse(Rule::rules, input).unwrap_or_else(|e| panic!("{}", e));
        return SystemTemplate::evaluate(pairs, values, functions); // skip EOI (last entry in first pairs)
    }

    fn evaluate(pairs: Pairs<Rule>, values: &HashMap<String, VariableValue>, functions: &HashMap<String, Box<dyn FunctionEvaluator>>) -> String {
        let mut words: Vec<String> = Vec::new();
        for pair in pairs.into_iter() {
            match pair.as_rule() {
                Rule::text => { words.push(pair.as_str().to_string()); },
                Rule::addition => { 
                    let result = SystemTemplate::eval_addition(pair, values, functions);
                    match result {
                        VariableValue::BoolVar(content) => { words.push( String::from(if content == true { "WAHR" } else { "FALSCH" })); },
                        VariableValue::FloatVar(content) => { words.push( content.to_string() ); },
                        VariableValue::StringVar(content) => { words.push(content); },
                    }
                },
                Rule::EOI => {},
                _ => { panic!("Unexpected rule {:?}, expected text or addition", pair.as_rule()); }
            }
        }
        return words.join(" ");
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
            return SystemTemplate::evaluate_multiplication(inner.pop().unwrap(), values, functions);
        } else {
            // len >= 3 and we have integers here, otherwise panic since we would be trying to add e.g. strings
            let mut result = 0.0;
            let mut operation = String::from("+");
            for rule in inner.into_iter() {
                match rule.as_rule() {
                    Rule::multiplication => { 
                        if let VariableValue::FloatVar(rhs) = SystemTemplate::evaluate_multiplication(rule.clone(), values, functions) {
                            result = SystemTemplate::perform_operation(result, rhs, &operation);
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
                            Rule::addition => { args.push(SystemTemplate::eval_addition(fn_arg, values, functions)); },
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
                Rule::addition => { return SystemTemplate::eval_addition(inner.pop().unwrap(), values, functions); },
                Rule::function => { return SystemTemplate::evaluate_function(inner.pop().unwrap(), values, functions); },
                _ => { return SystemTemplate::eval_token(inner.pop().unwrap(), values); },
            }
        } else {
            // len >= 3 and we have integers here, otherwise panic since we would be trying to add e.g. strings
            let mut result = 1.0;
            let mut operation = String::from("*");
            for rule in inner.into_iter() {
                println!(" RULE: {:?}", rule);
                match rule.as_rule() {
                    Rule::addition => {
                        if let VariableValue::FloatVar(rhs) = SystemTemplate::eval_addition(rule.clone(), values, functions) {
                            result = SystemTemplate::perform_operation(result, rhs, &operation);
                        } else {
                            panic!("Couldn't convert rhs to f32 {:?}: {:?}", rule.as_rule(), rule.as_str());
                        }
                    },
                    Rule::multiply | Rule::divide => { operation = rule.as_str().to_string(); },
                    Rule::number | Rule::negative_number => { 
                        if let VariableValue::FloatVar(rhs) = SystemTemplate::eval_token(rule.clone(), values) {
                            result = SystemTemplate::perform_operation(result, rhs, &operation);
                        } else {
                            panic!("Couldn't convert rhs to f32 {:?}: {:?}", rule.as_rule(), rule.as_str());
                        }
                    },
                    Rule::identifier => {
                        if let VariableValue::FloatVar(val) = values.get(rule.as_str()).unwrap() {
                            result =  SystemTemplate::perform_operation(result, val.clone(), &operation);
                        } else {
                            panic!("Unexpected datatype in rule {:?}", rule);
                        }
                    },
                    _ => {panic!("Unexpected multiplication rule {:?} with content {:?}", rule.as_rule(), rule.as_str()); },
                }
            }
            return VariableValue::FloatVar(result);
        }
    }

    
}



#[cfg(test)]
mod tests {
    use crate::parsers::system_parser::*;

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
    fn test_system_variable_finder() {
        let mut functions: HashMap<String, Box<dyn FunctionEvaluator>> = HashMap::new();
        functions.insert(String::from("TABLE1.FUNCTION1"), Box::new(TestFunction{}));
        let test_str = "Test {{TABLE1.FUNCTION1(1,2,3,4) }} MIT wert {{ VAR1 * 2 }}!";
        let variables = SystemTemplate::find_variables(test_str);
        assert_eq!(variables.len(),1);
        let var1 = variables.get("VAR1");
        assert_ne!(var1, None);
        assert_eq!(var1.unwrap(), "VAR1");
    }

    #[test]
    fn test_system_function_evaluation() {
        let mut functions: HashMap<String, Box<dyn FunctionEvaluator>> = HashMap::new();
        functions.insert(String::from("TABLE1.FUNCTION1"), Box::new(TestFunction{}));
        let mut values: HashMap<String, VariableValue> = HashMap::new();
        values.insert(String::from("VAR1"), VariableValue::FloatVar(0.5));
       
        let test_str = "Test!";
        let result = SystemTemplate::evaluate_template(test_str, &values, &functions);
        assert_eq!(result, "Test!");

        let test_str = "Test! MIT wert {{ VAR1 * 2 }}!";
        let result = SystemTemplate::evaluate_template(test_str, &values, &functions);
        assert_eq!(result, "Test! MIT wert 1 !");
        
        let test_str = "Test! MIT wert {{ VAR1 * 2 }} und Funktion {{ TABLE1.FUNCTION1(1,2,3,4) }}!";
        let result = SystemTemplate::evaluate_template(test_str, &values, &functions);
        assert_eq!(result, "Test! MIT wert 1 und Funktion 10 !");
    }

}