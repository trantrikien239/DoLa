### TruthfulQA (Multiple Choices)
Re-run the evaluation for Llama-1 and Llama-2

```bash
# Llama 1 baseline
!cd .. && python tfqa_mc_eval.py --model-name huggyllama/llama-7b --data-path ./tmp/ --output-path ../experiments/output-path-tfmc-baseline-llama-1.json --num-gpus 1
# Llama 2 baseline
!cd .. && python tfqa_mc_eval.py --model-name TheBloke/Llama-2-7B-fp16 --data-path ./tmp/ --output-path ../experiments/output-path-tfmc-baseline-llama-2.json --num-gpus 1
# Llama 1 DoLA
!cd .. && python tfqa_mc_eval.py --model-name huggyllama/llama-7b --early-exit-layers 16,18,20,22,24,26,28,30,32 --data-path ./tmp/ --output-path ../experiments/output-path-tfqamc-dola-llama-1.json --num-gpus 1
# Llama 2 DoLA
!cd .. && python tfqa_mc_eval.py --model-name TheBloke/Llama-2-7B-fp16 --early-exit-layers 16,18,20,22,24,26,28,30,32 --data-path ./tmp/ --output-path ../experiments/output-path-tfqamc-dola-llama-2.json --num-gpus 1
```
