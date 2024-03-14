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
        
    # def test_instantiate(self):
    #     self.assertTrue(self.trainer, 'Trainer is None')
        
    # def test_query_expand(self):
    #     # how to pass queries into the new_trainer
    #     queries = ['My name is', 'The cat chased']
    #     query_ids = self.tokenizer(queries, return_tensors='pt')
    #     query_ids['input_ids'] = query_ids['input_ids']
    #     query_ids['attention_mask'] = query_ids['attention_mask']
        
    #     num_to_explore = 3
        
    #     resp = self.trainer.explore_expand_tensors(query_tensor=query_ids['input_ids'], num_to_explore=num_to_explore)
    #     self.assertEqual(len(resp), 2*num_to_explore, 
    #                      f"Expected length {2*num_to_explore}. Got {len(resp)}")
    #     self.assertEqual(list(resp[0].shape), [len(query_ids['input_ids'][0])], 
    #                      f"Expected item size {len(query_ids['input_ids'][0])}. Got {list(resp[0].shape)}")
    
    @unittest.skip('Manual Override')
    def test_generate_batched(self):
        queries = ['My name is', 'The cat chased']
        query_ids = self.tokenizer(queries, return_tensors='pt')
        query_ids['input_ids'] = query_ids['input_ids']
        query_ids['attention_mask'] = query_ids['attention_mask']
        num_to_explore=2
        generation_kwargs = {
            # "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": 100_000,
            "remove_padding": False,
            "return_prompt": False,
            "pad_token_id": self.tokenizer.pad_token_id,
            "max_new_tokens": 64,
        }
        resp = self.trainer.generate(query_tensor=query_ids['input_ids'], num_to_explore=num_to_explore, **generation_kwargs)
        for r in resp:
            self.assertEqual(r.shape[0], generation_kwargs['max_new_tokens']-1) # minus 1 is due to the eos token being removed.
            
    def test_model_used_in_batch_gen(self):
        queries = ['My name is', 'The cat chased']
        query_ids = self.tokenizer(queries, return_tensors='pt')
        query_ids['input_ids'] = query_ids['input_ids']
        query_ids['attention_mask'] = query_ids['attention_mask']
        
        padded_inputs = self.tokenizer.pad(
                query_ids,
                padding=True,
                max_length=None,
                pad_to_multiple_of=None,
                return_tensors="pt",
        ).to('cuda')
        
        generation_kwargs = {
            # "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": 100_000,
            # "remove_padding": False,
            # "return_prompt": False,
            "pad_token_id": self.tokenizer.pad_token_id,
            "max_new_tokens": 64,
            "return_dict_in_generate": True,
            "output_logits":True
        }
        output = self.trainer._accelerate_model(self.model, padded_inputs, **generation_kwargs)
        self.assertEqual(list(output.sequences.shape), [2, 4 + 64])
        for token_logits in list(output.logits):
            self.assertEqual(list(token_logits.shape), [2, 50272])
    
    @unittest.skip('Manual Override')
    def test_generate_batched_logits(self):
        queries = ['My name is', 'The cat chased']
        query_ids = self.tokenizer(queries, return_tensors='pt')
        query_ids['input_ids'] = query_ids['input_ids']
        query_ids['attention_mask'] = query_ids['attention_mask']
        num_to_explore=2
        generation_kwargs = {
            # "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": 100_000,
            "remove_padding": False,
            "return_prompt": False,
            "pad_token_id": self.tokenizer.pad_token_id,
            "max_new_tokens": 64,
            "return_dict_in_generate": True,
            "output_logits":True
            
        }
        resp = self.trainer.generate(query_tensor=query_ids['input_ids'], num_to_explore=num_to_explore, **generation_kwargs)
        # print(resp)
            
    
