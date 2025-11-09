# Goal
Design a small, reproducible project to evaluate an open model (e.g., Qwen2.5-7B-Instruct) served by SGLang on the AIME 2024 benchmark,         
reporting exact-answer accuracy.                                                                                                                                                                                                                                                              
  Repo La yout                                                                                                                                     
                                                                                                                    
  -GPT_OSS_TEST_BENCHMARK/                                                                                                                   
      - data/aime24.jsonl  — problems and gold answers (JSONL).                                                                                   
      - eval/run_aime.py   — queries SGLang and scores results.                                                                                   
      - prompts/system.txt — minimal system prompt.                                                                                               
      - results/           — saved predictions and summary.                                                                                       
      - requirements.txt   — sglang, openai, typer/argparse, tqdm.                                                                                
                                                                                                                                                  
  Setup                                                                                                                                           
                                                                                                                                                  
  - Recommended: Linux/WSL2 + CUDA GPU.                                                                                                           
  - Create env: python -m venv .venv && . .venv/bin/activate                                                                                      
  - Install: pip install -U "sglang[all]" openai tqdm                                                                                             
  - Prepare dataset file data/aime24.jsonl with rows like:                                                                                        
      - {"question_id": 1, "problem": "Compute 1+2+...+20.", "answer": 210}                                                                       
                                                                                                                                                  
  Serve Model (SGLang)                                                                                                                            
                                                                                                                                                  
  - Start server (example: Qwen2.5-7B-Instruct):                                                                                                  
      - python -m sglang.launcher --model Qwen/Qwen2.5-7B-Instruct --port 30000 --trust-remote-code                                               
  - Use OpenAI-compatible API:                                                                                                                    
      - export OPENAI_BASE_URL=http://127.0.0.1:30000/v1                                                                                          
      - export OPENAI_API_KEY=sk-noop (any non-empty value)                                                                                       
                                                                                                                                                  
  Evaluation Script (minimal)                                                                                                                     
                                                                                                                                                  
  - Create eval/run_aime.py:                                                                                                                      
      - Calls OpenAI Chat Completions at OPENAI_BASE_URL.                                                                                         
      - Prompt: “Give only the final integer answer (0–999). No steps.”                                                                           
      - Parser: prefer \boxed{n}; else Final Answer: n; else last 1–3 digit integer.                                                              
      - Computes accuracy and writes per-item results.                                                                                            
                                                                                                                                                  
  Example core (trimmed):                                                                                                                         
                                                                                                                                                  
  - from openai import OpenAI                                                                                                                     
  - client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL"), api_key=os.getenv("OPENAI_API_KEY","sk-noop"))                                         
  - resp = client.chat.completions.create(model=args.model, messages=[{"role":"system", "content":sys_prompt}, {"role":"user","content":q}],      
    temperature=0.0, max_tokens=256)                                                                                                              
  - pred = extract_aime_answer(resp.choices[0].message.content)                                                                                   
                                                                                                                                                  
  Run Benchmark                                                                                                                                   
                                                                                                                                                  
  - python eval/run_aime.py --model Qwen/Qwen2.5-7B-Instruct --dataset data/aime24.jsonl --out results/aime24_qwen7b.jsonl                        
  - Output prints Accuracy: X/Y = Z.ZZZ and saves raw predictions.                                                                                
                                                                                                                                                  
  Notes                                                                                                                                           
                                                                                                                                                  
  - AIME answers must be integers in [0,999]; reject or clip others.                                                                              
  - For larger models, add --tensor-parallel-size and adjust batch sizing.                                                                        
  - Want me to scaffold these files under GPT_OSS_TEST_BENCHMARK/ now? I’ll create the folders, example dataset row, prompt, and evaluator.