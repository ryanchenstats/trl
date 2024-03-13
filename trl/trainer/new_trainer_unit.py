import unittest
from trl.trainer.new_trainer import NEWTrainer
from trl import PPOConfig
from trl import AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer, AutoModelForCausalLM

class TestNewTrainer(unittest.TestCase):
    
    def setUp(self) -> None:
        self.trainer = None
        self.tokenizer = AutoTokenizer.from_pretrained('facebook/opt-350m')
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained('facebook/opt-350m', cache_dir = 'unittest/')
        
        ppo_config = PPOConfig(
            model_name='facebook/opt-350m',
            learning_rate=1.41e-5,
            batch_size=64,
            mini_batch_size=4,
            gradient_accumulation_steps=4
            # log_with="wandb",
        )
        
        self.trainer = NEWTrainer(config=ppo_config, model=self.model, tokenizer=self.tokenizer)
        return super().setUp()
        
    def test_instantiate(self):
        self.assertTrue(self.trainer, 'Trainer is None')
        
    def test_query_expand(self):
        # how to pass queries into the new_trainer
        queries = ['My name is', 'The cat chased']
        query_ids = self.tokenizer(queries, return_tensors='pt')
        query_ids['input_ids'] = query_ids['input_ids']
        query_ids['attention_mask'] = query_ids['attention_mask']
        
        num_to_explore = 3
        
        resp = self.trainer.explore_expand_tensors(query_tensor=query_ids['input_ids'], num_to_explore=num_to_explore)
        self.assertEqual(list(resp.shape), [2*num_to_explore, len(query_ids['input_ids'][0])], 
                         f'Expected {[2*num_to_explore, len(query_ids["input_ids"][0])]}. Got {resp.shape}')
        
    def test_generate_batched(self):
        queries = ['My name is', 'The cat chased']
        query_ids = self.tokenizer(queries, return_tensors='pt')
        query_ids['input_ids'] = query_ids['input_ids']
        query_ids['attention_mask'] = query_ids['attention_mask']
        num_to_explore=2
                
        resp = self.trainer.generate(query_tensor=query_ids['input_ids'], num_to_explore=num_to_explore)
        pass