use std::{path::Path, collections::HashMap};
use std::fs;

use serde::{Serialize, Deserialize};
use serde_aux::prelude::*;

#[derive(Debug, Eq, PartialEq, Hash, Serialize, Deserialize, Clone)]
pub struct JSONAnswer {
    #[serde(deserialize_with = "deserialize_number_from_string")]
    pub id: u64,
    #[serde(deserialize_with = "deserialize_number_from_string", rename(deserialize = "nodeId"))]
    pub node_id: u64,
    pub raw_text: String,
}

#[derive(Debug, Eq, PartialEq, Hash, Serialize, Deserialize, Clone)]
pub struct JSONFAQQuestion {
    #[serde(deserialize_with = "deserialize_number_from_string")]
    pub id: u64,
    #[serde(deserialize_with = "deserialize_number_from_string", rename(deserialize = "nodeId"))]
    pub node_id: u64,
    pub text: String,
}

#[derive(Debug, Eq, PartialEq, Hash, Serialize, Deserialize, Clone)]
pub struct JSONNodeData {
    pub raw_text: String,
    #[serde(rename(deserialize = "type"))]
    pub node_type: String,
    #[serde(deserialize_with = "deserialize_number_from_string")]
    pub id: u64,
    pub answers: Vec<JSONAnswer>,
    pub questions: Vec<JSONFAQQuestion>,
}

#[derive(Debug, Eq, PartialEq, Hash, Serialize, Deserialize, Clone)]
pub struct JSONNode {
    pub data: JSONNodeData,
}

#[derive(Debug, Eq, PartialEq, Hash, Serialize, Deserialize, Clone)]
pub struct JSONConnection {
    #[serde(deserialize_with = "deserialize_number_from_string")]
    pub id: u64,      
    #[serde(deserialize_with = "deserialize_number_from_string", rename(deserialize = "source"))]
    pub source_node_id: u64,  
    #[serde(deserialize_with = "deserialize_number_from_string", rename(deserialize = "sourceHandle"))]
    pub source_answer_id: u64,
    #[serde(deserialize_with = "deserialize_number_from_string", rename(deserialize = "target"))]
    pub target_node_id: u64,
}

#[derive(Debug, Eq, PartialEq, Hash, Serialize, Deserialize, Clone)]
pub struct JSONData {
    pub nodes: Vec<JSONNode>,
    pub connections: Vec<JSONConnection>,
}


pub fn load_json(path: &str) -> JSONData {
    let json_file_path = Path::new(path);
    let data_str = fs::read_to_string(json_file_path).expect("couldn't read file");
    return serde_json::from_str(&data_str).unwrap();
}


pub fn load_answer_synonyms(path: &str) -> HashMap<String, Vec<String>> {
    let json_file_path = Path::new(path);
    let data_str = fs::read_to_string(json_file_path).expect("couldn't read file");
    return serde_json::from_str(&data_str).unwrap();
}


#[cfg(test)]
mod tests {
    use crate::data::json::load_json;

    #[test]
    fn test_load_json() {
        let path = "/fs/scratch/users/vaethdk/adviser_reisekosten/train_graph.json";
        let data = load_json(path);
        assert_eq!(data.nodes.len(), 127);
    }
}
