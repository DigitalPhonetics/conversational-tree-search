use std::collections::HashSet;
use std::rc::Rc;
use std::{collections::HashMap};

use super::json::{load_json, JSONConnection, load_answer_synonyms};

#[derive(Debug, Clone)]
pub enum DatasetMode {
    TRAIN,
    TEST,
}

#[derive(Debug, Clone)]
pub struct DialogAnswer {
    pub id: u64,
    pub text: String,
    pub connected_node_id: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct FreeformQuestion {
    pub id: u64,
    pub node_id: u64,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum NodeType {
    Start,
    Information,
    Question,
    Variable,
    Logic,
}

#[derive(Debug, Clone)]
pub struct DialogNode {
    pub id: u64,
    pub node_type: NodeType,
    pub connected_node_id: Option<u64>,
    pub text: String,
    pub answers: Vec<Rc<DialogAnswer>>,
    pub questions: Vec<Rc<FreeformQuestion>>,
}


pub struct Dataset {
    pub mode: DatasetMode,

    node_by_id: HashMap<u64, Rc<DialogNode>>,
    answer_by_id: HashMap<u64, Rc<DialogAnswer>>,

    start_node: Rc<DialogNode>,

    questions: Vec<Rc<FreeformQuestion>>,
    answer_synonyms: HashMap<String, Vec<String>>,

    pub max_node_degree: usize,
    pub max_tree_depth: u8,
}

fn get_max_tree_depth(current_node_id: u64, visited_ids: &mut HashSet<u64>, node_by_id: &HashMap<u64, Rc<DialogNode>>, answer_by_id: &HashMap<u64, Rc<DialogAnswer>>) -> u8 {
    // start with current_depth = 0, current_node_id = 0 (startNode)

    // break loops
    if visited_ids.contains(&current_node_id) {
        return 0;
    }
    visited_ids.insert(current_node_id);

    // follow children
    let current_node = node_by_id.get(&current_node_id).unwrap();
    if let Some(child_id) = current_node.connected_node_id {
        // direct connection to neighbour
        return 1 + get_max_tree_depth(child_id, visited_ids, node_by_id, answer_by_id)
    } else if current_node.answers.len() > 0 {
        // follow all children, return max
        return 1 + current_node.answers.iter().map(|answer| 
            if answer.connected_node_id.is_some() { get_max_tree_depth(answer.connected_node_id.unwrap(), visited_ids, node_by_id, answer_by_id) } else {0}
        ).max().unwrap();
    } else {
        // reached leaf node
        return 1;
    }
}


impl Dataset {
    fn get_connected_node(&self, node: &DialogNode) -> Option<&Rc<DialogNode>> {
        return match node.connected_node_id {
            Some(id) => self.node_by_id.get(&id).clone(),
            _ => None,
        }
    }
    
    fn get_answer_target(&self, answer: &DialogAnswer) -> Option<&Rc<DialogAnswer>> {
        return match answer.connected_node_id {
            Some(id) => self.answer_by_id.get(&id).clone(),
            _ => None,
        }
    }

    fn get_start_node(&self) -> &Rc<DialogNode> {
        return self.node_by_id.get(&self.start_node.connected_node_id.unwrap()).unwrap();
    }

    fn new(mode: DatasetMode) -> Self {
        // load json data
        let data_path = match mode {
            DatasetMode::TRAIN => "/fs/scratch/users/vaethdk/adviser_reisekosten/train_graph.json",
            DatasetMode::TEST => "/fs/scratch/users/vaethdk/adviser_reisekosten/test_graph.json",
        };
        let json_data = load_json(data_path);
        
        let synonym_path = match mode {
            DatasetMode::TRAIN => "/fs/scratch/users/vaethdk/adviser_reisekosten/train_answers.json",
            DatasetMode::TEST => "/fs/scratch/users/vaethdk/adviser_reisekosten/test_answers.json",
        };
        let answer_synonyms = load_answer_synonyms(synonym_path);

        // transform json data into Dataset
        let mut max_node_degree: usize = 0;
        let mut node_by_id = HashMap::new();
        let mut answer_by_id = HashMap::new();
        let mut questions = Vec::new();

        // connections
        let connection_by_answer_id: HashMap<u64, JSONConnection> = json_data
            .connections
            .into_iter()
            .map(|conn| return (conn.source_answer_id, conn))
            .collect();

        // nodes
        let mut start_node: Option<Rc<DialogNode>> = None;
        for json_node in json_data.nodes.iter() {
            let node_type = match json_node.data.node_type.as_str() {
                "startNode" => NodeType::Start,
                "infoNode" => NodeType::Information,
                "userInputNode" => NodeType::Variable,
                "userResponseNode" => NodeType::Question,
                "logicNode" => NodeType::Logic,
                _ => {
                    panic!("Unknown node type: {}", json_node.data.node_type)
                }
            };

            let node_questions: Vec<Rc<FreeformQuestion>> = json_node.data.questions.iter().map(|q| 
                Rc::new(FreeformQuestion {
                    id: q.id,
                    node_id: q.node_id,
                    text: q.text.clone(),
                })
            ).collect();
            for question in node_questions.iter() {
                questions.push(question.clone());
            }

            let node_answers: Vec<Rc<DialogAnswer>> = json_node.data.answers.iter().map(|a| 
                Rc::new(DialogAnswer {
                    id: a.id,
                    text: a.raw_text.clone(),
                    connected_node_id: connection_by_answer_id.get(&a.id).and_then(|conn| Some(conn.target_node_id)),
                })
            ).collect();
            for answer in node_answers.iter() {
                answer_by_id.insert(answer.id, answer.clone());
            }

            let mut connected_node_id = None;
            if let Some(conn) = connection_by_answer_id.get(&json_node.data.id) {
                if conn.source_answer_id == conn.source_node_id {
                    connected_node_id = Some(conn.target_node_id);
                }
            }

            let node = Rc::new(DialogNode {
                id: json_node.data.id,
                node_type: node_type.clone(),
                connected_node_id: connected_node_id,
                text: json_node.data.raw_text.clone(),
                answers: node_answers,
                questions: node_questions,
            });
            if node.answers.len() > max_node_degree {
                max_node_degree = node.answers.len();
            }
            node_by_id.insert(json_node.data.id, node.clone());
            if node_type == NodeType::Start {
                start_node = Some(node);
            }
        }
        assert!(start_node.is_some(), "Start node is empty");

        // get max tree depth
        let mut visited_ids: HashSet<u64> = HashSet::new();
        let start_node_id = start_node.clone().unwrap().connected_node_id.unwrap();
        let max_tree_depth = get_max_tree_depth(start_node_id, &mut visited_ids, &node_by_id, &answer_by_id);

        return Dataset {
            mode: mode,
            start_node: start_node.unwrap(),
            node_by_id: node_by_id,
            answer_by_id: answer_by_id,
            questions: questions,
            answer_synonyms: answer_synonyms,
            max_node_degree: max_node_degree,
            max_tree_depth: max_tree_depth,
        };
    }
}


#[cfg(test)]
mod tests {
    use super::Dataset;

    #[test]
    fn test_load_train_graph() {
        let ds = Dataset::new(super::DatasetMode::TRAIN);
        assert_eq!(ds.node_by_id.len(), 127);
        assert_eq!(ds.start_node.id, 0);
        assert_eq!(ds.get_start_node().answers.len(), 6);
        println!("Tree depth: {:?}", ds.max_tree_depth);
        println!("Max node degree: {:?}", ds.max_node_degree);
    }


    #[test]
    fn test_load_test_graph() {
        let ds = Dataset::new(super::DatasetMode::TEST);
        assert_eq!(ds.node_by_id.len(), 127);
        assert_eq!(ds.start_node.id, 0);
        assert_eq!(ds.get_start_node().answers.len(), 6);
        println!("Tree depth: {:?}", ds.max_tree_depth);
        println!("Max node degree: {:?}", ds.max_node_degree);
    }
    
}