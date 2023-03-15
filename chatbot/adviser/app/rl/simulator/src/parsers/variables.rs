use std::{collections::{HashMap}, borrow::{Borrow}, fs};
use rand::{seq::{SliceRandom, IteratorRandom}, Rng};

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum VariableValue {
    StringVar(String),
    BoolVar(bool),
    FloatVar(f32),
}

#[derive(Debug, PartialEq)]
pub enum VariableType {
    TEXT,
    NUMBER,
    TIMEPOINT,
    TIMESPAN,
    BOOLEAN,
    LOCATION
}

pub trait FunctionEvaluator {
    fn evaluate(&self, args: &Vec<VariableValue>) -> VariableValue;
}

pub enum Comperator {
    LT,
    LEQ,
    EQ,
    NEQ,
    GEQ,
    GT,
}

pub struct VariableConstraint {
    comperator: Comperator,
    value: VariableValue,
}

pub struct ConstrainedVariableValue {
    pub var_name: String,
    pub var_type: VariableType,

    lt_condition: Option<VariableValue>,
    leq_condition: Option<VariableValue>,
    eq_condition: Option<VariableValue>,
    neq_conditions: Vec<VariableValue>,
    gt_condition: Option<VariableValue>,
    geq_condition: Option<VariableValue>,
}


pub struct LocationSynonyms {
    pub countries: HashMap<String, Vec<String>>,
    pub cities: HashMap<String, Vec<String>>,
}

impl LocationSynonyms {
    pub fn new() -> Self {
        let data = fs::read_to_string("./parser/country_synonyms.json").expect("Unable to read file");
        let countries = serde_json::from_str(&data).expect("JSON does not have correct format.");

        let data = fs::read_to_string("./parser/city_synonyms.json").expect("Unable to read file");
        let cities = serde_json::from_str(&data).expect("JSON does not have correct format.");

        return LocationSynonyms { countries: countries, cities: cities };
    }
}

fn flip_comperator(comperator : &Comperator) -> Comperator {
    return match comperator {
        Comperator::EQ => Comperator::NEQ,
        Comperator::NEQ => Comperator::EQ,
        Comperator::LT => Comperator::GEQ,
        Comperator::LEQ => Comperator::GT,
        Comperator::GEQ => Comperator::LT,
        Comperator::GT => Comperator::LEQ,
    };
}

fn compare(condition: &Option<VariableValue>, comperator: Comperator, rhs: &VariableValue) -> bool {
    if let Some(val) = condition {
        return match comperator {
            Comperator::EQ => val == rhs,
            Comperator::GEQ => val >= rhs,
            Comperator::GT => val > rhs,
            Comperator::LEQ => val <= rhs,
            Comperator::LT => val < rhs,
            Comperator::NEQ => val != rhs,
        }
    }
    return true;
}

impl ConstrainedVariableValue {
    pub fn new(var_name: String, var_type: VariableType) -> Self {
        return ConstrainedVariableValue { 
            var_name: var_name,
            var_type: var_type, 
            lt_condition: None, 
            leq_condition: None,
            eq_condition: None,
            neq_conditions: Vec::new(),
            gt_condition: None,
            geq_condition: None
        }
    }

    fn draw_number(&self) -> f32 {
        let mut lower_bound: f32 = 0.0;
        if let Some(VariableValue::FloatVar(gt_val)) = self.gt_condition {
            // set lower bound to highest condition if both are set
            if let Some(VariableValue::FloatVar(geq_val)) = self.geq_condition {
                lower_bound = gt_val.max(geq_val - 1.0);
            } else {
                lower_bound = gt_val;
            }
        } else if let Some(VariableValue::FloatVar(geq_val)) = self.geq_condition {
            lower_bound = geq_val;
        }

        let mut upper_bound: f32 = 0.0;
        if let Some(VariableValue::FloatVar(lt_val)) = self.lt_condition {
            // set lower bound to highest condition if both are set
            if let Some(VariableValue::FloatVar(leq_val)) = self.leq_condition {
                upper_bound = lt_val.min(leq_val + 1.0);
            } else {
                upper_bound = lt_val;
            }
        } else if let Some(VariableValue::FloatVar(leq_val)) = self.leq_condition {
            upper_bound = leq_val;
        } else {
            if lower_bound != 0.0 {
                upper_bound = 100.0 * lower_bound; 
            } else {
                upper_bound = 10000.0;
            }
        }
        
        let mut rng = rand::thread_rng();
        return rng.gen_range(lower_bound..upper_bound);
    }

    pub fn draw_value(&self, locations: &LocationSynonyms) -> VariableValue {
        if let Some(value) = &self.eq_condition {
            return value.clone();
        }

        return match self.var_type {
            VariableType::BOOLEAN => {
                if let Some(val) = &self.eq_condition {
                    return val.clone();
                } else if self.neq_conditions.len() > 0 {
                    if let VariableValue::BoolVar(value) = self.neq_conditions.iter().next().unwrap() {
                        return VariableValue::BoolVar(value.clone());
                    }
                }
                let res = vec![true, false].choose(&mut rand::thread_rng()).unwrap().clone();
                return VariableValue::BoolVar(res);
            },
            VariableType::LOCATION => {
                if self.var_name.to_lowercase().contains("land") {
                    // draw country
                    let mut country = locations.countries.keys().choose(&mut rand::thread_rng()).unwrap();
                    while self.neq_conditions.contains(&VariableValue::StringVar(country.clone())) {
                        country = locations.countries.keys().choose(&mut rand::thread_rng()).unwrap(); 
                    }
                    return VariableValue::StringVar(country.clone());
                } else {
                    // draw city
                    let mut city = locations.cities.keys().choose(&mut rand::thread_rng()).unwrap();
                    while self.neq_conditions.contains(&VariableValue::StringVar(city.clone())) {
                        city = locations.cities.keys().choose(&mut rand::thread_rng()).unwrap(); 
                    }
                    return VariableValue::StringVar(city.clone());
                }
            },
            VariableType::TEXT => todo!(),
            VariableType::NUMBER => VariableValue::FloatVar(self.draw_number()),
            VariableType::TIMEPOINT => VariableValue::FloatVar(self.draw_number()),
            VariableType::TIMESPAN => todo!(),
        }
    }


    /// adds a constraint and returns true if successful, false if the value violates existing constraints 
    pub fn add_condition(&mut self, comperator: Comperator, value: VariableValue) -> bool {
        let mut result = true;
        match comperator {
            Comperator::EQ => {
                result = result && compare(&self.eq_condition, Comperator::EQ, &value);
                result = result && self.neq_conditions.iter().all(|neq_val| neq_val != &value);
                result = result && compare(&self.lt_condition, Comperator::LT, &value);
                result = result && compare(&self.leq_condition, Comperator::LEQ, &value);
                result = result && compare(&self.gt_condition, Comperator::GT, &value);
                result = result && compare(&self.geq_condition, Comperator::GEQ, &value);
                self.eq_condition = Some(value);
            },
            Comperator::NEQ => {
                result = result && compare(&self.eq_condition, Comperator::NEQ, &value);
                self.neq_conditions.push(value.clone());
            },
            Comperator::LT => {
                result = result && compare(&self.eq_condition, Comperator::LT, &value);     // eq = a < new lt
                if let Some(val) = &self.lt_condition {
                    if val.borrow() > &value {
                        // change bounds to lower value
                        self.lt_condition = Some(value.clone()); 
                    }
                }
                result = result && compare(&self.gt_condition, Comperator::LT, &value);     // gt < a < new lt
                result = result && compare(&self.geq_condition, Comperator::LT, &value);   // geq <= a < new lt
                if self.lt_condition == None {
                    self.lt_condition = Some(value.clone());
                }
            },
            Comperator::LEQ => {
                result = result && compare(&self.eq_condition, Comperator::LEQ, &value);    // eq = a <= new leq
                if let Some(val) = &self.leq_condition {
                    if val.borrow() > &value {
                        self.leq_condition = Some(value.clone());
                    }
                }
                result = result && compare(&self.gt_condition, Comperator::LT, &value);     // gt < a <= new leq
                result = result && compare(&self.geq_condition, Comperator::LEQ, &value);    // geq <= a <=  new leq
                if self.leq_condition == None {
                    self.leq_condition = Some(value.clone());
                }
            },
            Comperator::GT => {
                result = result && compare(&self.eq_condition, Comperator::GT, &value);    // eq = a > new gt
                if let Some(val) = &self.gt_condition {
                    if val.borrow() < &value {
                        self.gt_condition = Some(value.clone());
                    }
                }
                result = result && compare(&self.lt_condition, Comperator::GT, &value);  // lt > a > new gt
                result = result && compare(&self.leq_condition, Comperator::GT, &value);     // leq >= a > gt
                if self.gt_condition == None {
                    self.gt_condition = Some(value.clone());
                }
            },
            Comperator::GEQ => {
                result = result && compare(&self.eq_condition, Comperator::GEQ, &value); // eq = a >= geq
                if let Some(val) = &self.geq_condition {
                    if val.borrow() < &value {
                        self.geq_condition = Some(value.clone());
                    }
                }
                result = result && compare(&self.lt_condition, Comperator::GT, &value);    // lt > a >= geq
                result = result && compare(&self.leq_condition, Comperator::LEQ, &value);       // leq >= a >= geq
            },
        }
        return result;
    }

    /// adds a default condition w.r.t. all other conditions, meaning default evaluates to true if and only if all other conditions evaluate to false
    pub fn add_default_condition(&mut self, other_conditions: Vec<VariableConstraint>) -> bool {
        // invert conditions in same branch statement (DEFAULT is always == condition)
        //  TODO: this is not sufficient, e.g. there could be multiple != statements and DEFAULT could trigger one of them
        let mut result = other_conditions.iter().map(|condition| {
            // add all other branches
            let value = condition.value.clone();
            let comperator = flip_comperator(&condition.comperator);
            self.add_condition(comperator, value)
        });
        // check if all branches were added without violations
        return result.all(|success| success == true);
    }
}