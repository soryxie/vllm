## Context Parallelism
```bash
 VLLM_ATTENTION_BACKEND=RING_FLASH_ATTN VLLM_LOGGING_LEVEL=INFO uv run vllm serve /home/ch/model/Llama-3.1-8B-Instruct --context-parallel-size 2 --enforce-eager --no-enable-prefix-caching
 ```

### Parameters
VLLM_ATTENTION_BACKEND=RING_FLASH_ATTN设置使用 ringattention
--context-parallel-size
--enforce-eager #目前不支持 cuda graph，必须关闭cuda graph