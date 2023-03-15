# adviser_reisekosten

1. Install requirements into your python environment: `pip install -r requirements.txt`
2. Data is in `train_graph.json` and `test_graph.json`
3. If training multiple models, you can start and enable a redis cache in `train_chatbot.py` and then execute `sh start_cache.sh`
4. To train a policy, change the configuration as needed in `train_chatbot.py` and execute `python train_chatbot.py` (requires GPU - make sure to deactivate `caching` options in configuration if run without redis) 

