 ```bash
 % python3 train.py

Note: Environment variable`HF_TOKEN` is set and is the current active token independently from the token you've just configured.
Successfully logged in to Hugging Face.
DeepSeekV3ForCausalLM(
  (model): DeepSeekV3Model(
    (embed_tokens): Embedding(49152, 384)
    (layers): ModuleList(
      (0-23): 24 x DeepSeekV3Block(
        (input_layernorm): RMSNorm()
        (attention): DeepSeekV3Attention(
          (q_proj): Linear(in_features=384, out_features=384, bias=False)
          (k_proj): Linear(in_features=384, out_features=128, bias=False)
          (v_proj): Linear(in_features=384, out_features=128, bias=False)
          (o_proj): Linear(in_features=384, out_features=384, bias=False)
        )
        (post_attention_layernorm): RMSNorm()
        (mlp): DeepSeekV3MLP(
          (moe): DeepSeekMoE(
            (experts): ModuleList(
              (0-7): 8 x DeepSeekExpertLayer(
                (gate_proj): Linear(in_features=384, out_features=1024, bias=False)
                (up_proj): Linear(in_features=384, out_features=1024, bias=False)
                (down_proj): Linear(in_features=1024, out_features=384, bias=False)
                (act_fn): SiLU()
              )
            )
            (router): Linear(in_features=384, out_features=8, bias=False)
          )
        )
      )
    )
    (norm): RMSNorm()
  )
  (lm_head): Linear(in_features=384, out_features=49152, bias=False)
)
Model parameters: 254896512
Using device: mps
Resolving data files: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 104/104 [00:00<00:00, 131.69it/s]
Resolving data files: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 104/104 [00:00<00:00, 444656.08it/s]

Verifying data streaming...
Data loading successful
Training:   0%|                                                                                                                                                    | 0/10000 [00:00<?, ?it/s]Starting training...
/Users/user1/Documents/erav3_repos/session_15/model.py:755: UserWarning: Casting complex values to real discards the imaginary part (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/native/Copy.cpp:305.)
  freqs_cis = self.freqs_cis.to(device=hidden_states.device, dtype=hidden_states.dtype)

Step 0
Loss: 11.0624
LR: 0.000300
Unique content: 0
Metrics saved to checkpoints/metrics_0.pt
Training:   1%|█▎                                                                                                                                      | 100/10000 [01:53<3:12:05,  1.16s/it]
Step 100
Loss: 7.4222
LR: 0.000300
Unique content: 800
Metrics saved to checkpoints/metrics_100.pt
Training:   2%|██▋                                                                                                                                     | 200/10000 [04:18<4:00:04,  1.47s/it]
Step 200
Loss: 5.6819
LR: 0.000300
Unique content: 1599
Metrics saved to checkpoints/metrics_200.pt
Training:   3%|████                                                                                                                                    | 300/10000 [06:43<3:49:10,  1.42s/it]
Step 300
Loss: 3.8482
LR: 0.000300
Unique content: 2399
Metrics saved to checkpoints/metrics_300.pt
Training:   4%|█████▍                                                                                                                                  | 400/10000 [09:03<3:41:22,  1.38s/it]
Step 400
Loss: 2.2598
LR: 0.000300
Unique content: 3199
Metrics saved to checkpoints/metrics_400.pt
Training:   5%|██████▊                                                                                                                                 | 500/10000 [11:19<3:29:49,  1.33s/it]
Step 500
Loss: 1.2879
LR: 0.000300
Unique content: 3999
Metrics saved to checkpoints/metrics_500.pt

Generating with temperature: 0.8

Generated (500 steps):
<< Once upon a time  is it music it it it it they they they they carefully out created typically three still,,,,,,,,,,,,,,,,,,,, of thinking of light of of of of of of of of of >>

Generating with temperature: 0.8

Generated (500 steps):
<< In a galaxy far away  didn how her something getting imagine I I I I I I you " home they sure you they you you you you you you you you you you you you you you you you you you you stand type like harm really think you you you your you you >>
Checkpoint saved at step 500
Model and tokenizer saved to smollm2_model_step_500
Training:   6%|████████▏                                                                                                                               | 600/10000 [13:59<3:18:56,  1.27s/it]
Step 600
Loss: 1.0140
LR: 0.000300
Unique content: 4799
Metrics saved to checkpoints/metrics_600.pt
Training:   7%|█████████▌                                                                                                                              | 700/10000 [16:07<3:20:48,  1.30s/it]
Step 700
Loss: 0.7799
LR: 0.000300
Unique content: 5599
Metrics saved to checkpoints/metrics_700.pt
Training:   8%|██████████▉                                                                                                                             | 800/10000 [18:15<3:11:27,  1.25s/it]
Step 800
Loss: 0.5637
LR: 0.000300
Unique content: 6399
Metrics saved to checkpoints/metrics_800.pt
Training:   9%|████████████▏                                                                                                                           | 900/10000 [20:22<3:24:40,  1.35s/it]
Step 900
Loss: 0.2855
LR: 0.000300
Unique content: 7197
Metrics saved to checkpoints/metrics_900.pt
Training:  10%|█████████████▌                                                                                                                         | 1000/10000 [22:29<3:07:14,  1.25s/it]
Step 1000
Loss: 0.3682
LR: 0.000300
Unique content: 7997
Metrics saved to checkpoints/metrics_1000.pt

Generating with temperature: 0.8

Generated (1000 steps):
<< Once upon a time - time a a a a a a a a a a





































 >>

Generating with temperature: 0.8

Generated (1000 steps):
<< In a galaxy far away  practicing practicing three practicing as The,,,,,,,1Wb

































 >>
Checkpoint saved at step 1000
Model and tokenizer saved to smollm2_model_step_1000
Training:  11%|██████████████▊                                                                                                                        | 1100/10000 [24:55<3:06:00,  1.25s/it]
Step 1100
Loss: 0.3808
LR: 0.000300
Unique content: 8797
Metrics saved to checkpoints/metrics_1100.pt
Training:  12%|████████████████▏                                                                                                                      | 1200/10000 [27:00<3:04:54,  1.26s/it]
Step 1200
Loss: 0.3444
LR: 0.000300
Unique content: 9597
Metrics saved to checkpoints/metrics_1200.pt
Training:  13%|█████████████████▌                                                                                                                     | 1300/10000 [29:05<3:03:04,  1.26s/it]
Step 1300
Loss: 0.3715
LR: 0.000300
Unique content: 10396
Metrics saved to checkpoints/metrics_1300.pt
Training:  14%|██████████████████▉                                                                                                                    | 1400/10000 [31:12<3:00:29,  1.26s/it]
Step 1400
Loss: 0.1263
LR: 0.000300
Unique content: 11196
Metrics saved to checkpoints/metrics_1400.pt
Training:  15%|████████████████████▎                                                                                                                  | 1500/10000 [33:18<2:55:47,  1.24s/it]
Step 1500
Loss: 0.1435
LR: 0.000300
Unique content: 11995
Metrics saved to checkpoints/metrics_1500.pt

Generating with temperature: 0.8

Generated (1500 steps):
<< Once upon a time  let online too we remains,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,, >>

Generating with temperature: 0.8

Generated (1500 steps):
<< In a galaxy far away  it too getting sharing revealing revealing he he he he he he he he he he he he he he he he sometimes that that,,,,,,,,,,,,,,,,,,,,,,,,, >>
Checkpoint saved at step 1500
Model and tokenizer saved to smollm2_model_step_1500
Training:  16%|█████████████████████▌                                                                                                                 | 1600/10000 [35:44<2:49:24,  1.21s/it]
Step 1600
Loss: 0.0968
LR: 0.000300
Unique content: 12795
Metrics saved to checkpoints/metrics_1600.pt
Training:  17%|██████████████████████▉                                                                                                                | 1700/10000 [37:49<2:54:11,  1.26s/it]
Step 1700
Loss: 0.0603
LR: 0.000300
Unique content: 13594
Metrics saved to checkpoints/metrics_1700.pt
Training:  18%|████████████████████████▎                                                                                                              | 1800/10000 [39:57<2:57:18,  1.30s/it]
Step 1800
Loss: 0.0566
LR: 0.000300
Unique content: 14394
Metrics saved to checkpoints/metrics_1800.pt
Training:  19%|█████████████████████████▋                                                                                                             | 1900/10000 [42:04<2:48:57,  1.25s/it]
Step 1900
Loss: 0.0299
LR: 0.000300
Unique content: 15193
Metrics saved to checkpoints/metrics_1900.pt
Training:  20%|███████████████████████████                                                                                                            | 2000/10000 [44:09<2:47:01,  1.25s/it]
Step 2000
Loss: 0.0474
LR: 0.000300
Unique content: 15993
Metrics saved to checkpoints/metrics_2000.pt

Generating with temperature: 0.8

Generated (2000 steps):
<< Once upon a time  a clearly go hard things something someone someone someone someone someone someone someone she she that that that, that, never never never never never you you you you you you you you you you you you you you you you you you you you you you you you >>

Generating with temperature: 0.8

Generated (2000 steps):
<< In a galaxy far away  those if he sometimes sometimes sometimes sometimes do you you you you you you you you you you you you you knows knows knows knows knows knows knows seemed seemed seemed seemed seemed seemed seemed seemed seemed seemed seemed seemed seemed seemed seemed seemed seemed seemed seemed seemed seemed seemed >>
Checkpoint saved at step 2000
Model and tokenizer saved to smollm2_model_step_2000
Training:  21%|████████████████████████████▎                                                                                                          | 2100/10000 [46:35<2:43:14,  1.24s/it]
Step 2100
Loss: 0.0684
LR: 0.000300
Unique content: 16793
Metrics saved to checkpoints/metrics_2100.pt
Training:  22%|█████████████████████████████▋                                                                                                         | 2200/10000 [48:40<2:44:45,  1.27s/it]
Step 2200
Loss: 0.0594
LR: 0.000300
Unique content: 17593
Metrics saved to checkpoints/metrics_2200.pt
Training:  23%|███████████████████████████████                                                                                                        | 2300/10000 [50:45<2:39:07,  1.24s/it]
Step 2300
Loss: 0.0705
LR: 0.000300
Unique content: 18392
Metrics saved to checkpoints/metrics_2300.pt
Training:  24%|████████████████████████████████▍                                                                                                      | 2400/10000 [52:50<2:36:04,  1.23s/it]
Step 2400
Loss: 0.0644
LR: 0.000300
Unique content: 19190
Metrics saved to checkpoints/metrics_2400.pt
Training:  25%|█████████████████████████████████▊                                                                                                     | 2500/10000 [54:55<2:32:56,  1.22s/it]
Step 2500
Loss: 0.0396
LR: 0.000300
Unique content: 19989
Metrics saved to checkpoints/metrics_2500.pt

Generating with temperature: 0.8

Generated (2500 steps):
<< Once upon a time  a earlier they even whether whether whether whether whether whether whether if if - - - - - - - - - - - - - - - - -ingal education education in in in in inceptcept lead lead lead lead lead lead lead lead lead >>

Generating with temperature: 0.8

Generated (2500 steps):
<< In a galaxy far away  gets, did did off she I I I don don couldn couldn.,.,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,, >>
Checkpoint saved at step 2500
Model and tokenizer saved to smollm2_model_step_2500
Training:  26%|███████████████████████████████████                                                                                                    | 2600/10000 [57:18<2:29:48,  1.21s/it]
Step 2600
Loss: 0.0345
LR: 0.000300
Unique content: 20786
Metrics saved to checkpoints/metrics_2600.pt
Training:  27%|████████████████████████████████████▍                                                                                                  | 2700/10000 [59:21<2:30:53,  1.24s/it]
Step 2700
Loss: 0.0398
LR: 0.000300
Unique content: 21582
Metrics saved to checkpoints/metrics_2700.pt
Training:  28%|█████████████████████████████████████▏                                                                                               | 2800/10000 [1:01:24<2:26:55,  1.22s/it]
Step 2800
Loss: 0.0143
LR: 0.000300
Unique content: 22380
Metrics saved to checkpoints/metrics_2800.pt
Training:  29%|██████████████████████████████████████▌                                                                                              | 2900/10000 [1:03:28<2:26:35,  1.24s/it]
Step 2900
Loss: 0.0141
LR: 0.000300
Unique content: 23179
Metrics saved to checkpoints/metrics_2900.pt
Training:  30%|███████████████████████████████████████▉                                                                                             | 3000/10000 [1:05:33<2:24:26,  1.24s/it]
Step 3000
Loss: 0.0402
LR: 0.000300
Unique content: 23979
Metrics saved to checkpoints/metrics_3000.pt

Generating with temperature: 0.8

Generated (3000 steps):
<< Once upon a time  in hard in sure sure not not not not wrote which metadata$$$$$$$$$$60

).).).).00000000000000000000 >>

Generating with temperature: 0.8

Generated (3000 steps):
<< In a galaxy far away  was was was was was was was was now agreed agreedebical**:**:**:**:**:**:**:**:**:**:**:**:**:**:**:**:nnn terms risk walking risk also certain











 >>
Checkpoint saved at step 3000
Model and tokenizer saved to smollm2_model_step_3000
Training:  31%|█████████████████████████████████████████▏                                                                                           | 3100/10000 [1:07:56<2:18:13,  1.20s/it]
Step 3100
Loss: 0.0139
LR: 0.000300
Unique content: 24778
Metrics saved to checkpoints/metrics_3100.pt
Training:  32%|██████████████████████████████████████████▌                                                                                          | 3200/10000 [1:09:59<2:22:41,  1.26s/it]
Step 3200
Loss: 0.0202
LR: 0.000300
Unique content: 25576
Metrics saved to checkpoints/metrics_3200.pt
Training:  33%|███████████████████████████████████████████▉                                                                                         | 3300/10000 [1:12:03<2:21:01,  1.26s/it]
Step 3300
Loss: 0.0173
LR: 0.000300
Unique content: 26375
Metrics saved to checkpoints/metrics_3300.pt
Training:  34%|█████████████████████████████████████████████▏                                                                                       | 3400/10000 [1:14:08<2:15:45,  1.23s/it]
Step 3400
Loss: 0.0217
LR: 0.000300
Unique content: 27173
Metrics saved to checkpoints/metrics_3400.pt
Training:  35%|██████████████████████████████████████████████▌                                                                                      | 3500/10000 [1:16:11<2:11:44,  1.22s/it]
Step 3500
Loss: 0.0591
LR: 0.000300
Unique content: 27973
Metrics saved to checkpoints/metrics_3500.pt

Generating with temperature: 0.8

Generated (3500 steps):
<< Once upon a time - terms- evolved in in in in in in in in in in in in in in in0000000000000000000000000000111 >>

Generating with temperature: 0.8

Generated (3500 steps):
<< In a galaxy far away  around which which + + + + + +
       
       
       
       
       
       
       
       
       ....................... always might gather continuedAcAcAcAcAc >>
Checkpoint saved at step 3500
Model and tokenizer saved to smollm2_model_step_3500
Training:  36%|███████████████████████████████████████████████▉                                                                                     | 3600/10000 [1:18:35<2:10:23,  1.22s/it]
Step 3600
Loss: 0.0101
LR: 0.000300
Unique content: 28772
Metrics saved to checkpoints/metrics_3600.pt
Training:  37%|█████████████████████████████████████████████████▏                                                                                   | 3700/10000 [1:20:38<2:09:30,  1.23s/it]
Step 3700
Loss: 0.0057
LR: 0.000300
Unique content: 29570
Metrics saved to checkpoints/metrics_3700.pt
Training:  38%|██████████████████████████████████████████████████▌                                                                                  | 3800/10000 [1:22:42<2:07:09,  1.23s/it]
Step 3800
Loss: 0.0084
LR: 0.000300
Unique content: 30370
Metrics saved to checkpoints/metrics_3800.pt
Training:  39%|███████████████████████████████████████████████████▊                                                                                 | 3900/10000 [1:24:46<2:03:05,  1.21s/it]
Step 3900
Loss: 0.0147
LR: 0.000300
Unique content: 31170
Metrics saved to checkpoints/metrics_3900.pt
Training:  40%|█████████████████████████████████████████████████████▏                                                                               | 4000/10000 [1:26:49<2:04:01,  1.24s/it]
Step 4000
Loss: 0.0538
LR: 0.000300
Unique content: 31968
Metrics saved to checkpoints/metrics_4000.pt

Generating with temperature: 0.8

Generated (4000 steps):
<< Once upon a time  ( solving it it_.,. is is is is is is is is is is is,, + + + since + + + + + + revealing revealing revealing revealing revealing revealing revealing predict Studies Studies Studies Studies Studies Studies Studies Studies Studies Studies Studies >>

Generating with temperature: 0.8

Generated (4000 steps):
<< In a galaxy far away  more more more more more living800000000000000022222222222555555555in555555 ` >>
Checkpoint saved at step 4000
Model and tokenizer saved to smollm2_model_step_4000
Training:  41%|██████████████████████████████████████████████████████▌                                                                              | 4100/10000 [1:29:12<1:56:39,  1.19s/it]
Step 4100
Loss: 0.0207
LR: 0.000300
Unique content: 32766
Metrics saved to checkpoints/metrics_4100.pt
Training:  42%|███████████████████████████████████████████████████████▊                                                                             | 4200/10000 [1:31:12<1:55:22,  1.19s/it]
Step 4200
Loss: 0.0054
LR: 0.000300
Unique content: 33560
Metrics saved to checkpoints/metrics_4200.pt
Training:  43%|█████████████████████████████████████████████████████████▏                                                                           | 4300/10000 [1:33:13<1:53:03,  1.19s/it]
Step 4300
Loss: 0.0409
LR: 0.000300
Unique content: 34358
Metrics saved to checkpoints/metrics_4300.pt
Training:  44%|██████████████████████████████████████████████████████████▌                                                                          | 4400/10000 [1:35:16<1:55:09,  1.23s/it]
Step 4400
Loss: 0.0071
LR: 0.000300
Unique content: 35156
Metrics saved to checkpoints/metrics_4400.pt
Training:  45%|███████████████████████████████████████████████████████████▊                                                                         | 4500/10000 [1:37:19<1:50:08,  1.20s/it]
Step 4500
Loss: 0.0121
LR: 0.000300
Unique content: 35956
Metrics saved to checkpoints/metrics_4500.pt

Generating with temperature: 0.8

Generated (4500 steps):
<< Once upon a time  to discrimination similar looked by,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,, >>

Generating with temperature: 0.8

Generated (4500 steps):
<< In a galaxy far away , is,_______________________________________________ >>
Checkpoint saved at step 4500
Model and tokenizer saved to smollm2_model_step_4500
Training:  46%|█████████████████████████████████████████████████████████████▏                                                                       | 4600/10000 [1:39:42<1:48:29,  1.21s/it]
Step 4600
Loss: 0.0018
LR: 0.000300
Unique content: 36756
Metrics saved to checkpoints/metrics_4600.pt
Training:  47%|██████████████████████████████████████████████████████████████▌                                                                      | 4700/10000 [1:41:45<1:46:39,  1.21s/it]
Step 4700
Loss: 0.0042
LR: 0.000300
Unique content: 37554
Metrics saved to checkpoints/metrics_4700.pt
Training:  48%|███████████████████████████████████████████████████████████████▊                                                                     | 4800/10000 [1:43:50<1:46:10,  1.23s/it]
Step 4800
Loss: 0.0233
LR: 0.000300
Unique content: 38350
Metrics saved to checkpoints/metrics_4800.pt
Training:  49%|█████████████████████████████████████████████████████████████████▏                                                                   | 4900/10000 [1:45:53<1:40:08,  1.18s/it]
Step 4900
Loss: 0.0181
LR: 0.000300
Unique content: 39150
Metrics saved to checkpoints/metrics_4900.pt
Training:  50%|██████████████████████████████████████████████████████████████████▌                                                                  | 5000/10000 [1:47:56<1:41:00,  1.21s/it]
Step 5000
Loss: 0.0164
LR: 0.000300
Unique content: 39947
Metrics saved to checkpoints/metrics_5000.pt

Generating with temperature: 0.8

Generated (5000 steps):
<< Once upon a time 
 games - closer in in in in in in in in in in in in in in in in and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and >>

Generating with temperature: 0.8

Generated (5000 steps):
<< In a galaxy far away ions made she he he he he he he he he he:data):):):):):):):):):):):):):):):):==111111111111111111 >>
Checkpoint saved at step 5000
Model and tokenizer saved to smollm2_model_step_5000
Training:  51%|███████████████████████████████████████████████████████████████████▊                                                                 | 5100/10000 [1:50:19<1:38:38,  1.21s/it]
Step 5100
Loss: 0.0026
LR: 0.000300
Unique content: 40747
Metrics saved to checkpoints/metrics_5100.pt
Training:  52%|█████████████████████████████████████████████████████████████████████▏                                                               | 5200/10000 [1:52:22<1:39:17,  1.24s/it]
Step 5200
Loss: 0.0016
LR: 0.000300
Unique content: 41545
Metrics saved to checkpoints/metrics_5200.pt
Training:  53%|██████████████████████████████████████████████████████████████████████▍                                                              | 5300/10000 [1:54:24<1:37:06,  1.24s/it]
Step 5300
Loss: 0.0033
LR: 0.000300
Unique content: 42340
Metrics saved to checkpoints/metrics_5300.pt
Training:  54%|███████████████████████████████████████████████████████████████████████▊                                                             | 5400/10000 [1:56:27<1:35:47,  1.25s/it]
Step 5400
Loss: 0.0188
LR: 0.000300
Unique content: 43138
Metrics saved to checkpoints/metrics_5400.pt
Training:  55%|█████████████████████████████████████████████████████████████████████████▏                                                           | 5500/10000 [1:58:30<1:31:09,  1.22s/it]
Step 5500
Loss: 0.0008
LR: 0.000300
Unique content: 43934
Metrics saved to checkpoints/metrics_5500.pt

Generating with temperature: 0.8

Generated (5500 steps):
<< Once upon a time  to listening to Carta to to to to optimally optimally optimally rider rider Kevin Kevin Gl hosp hosp hosp hosp hospfourth sparks travelers more more needing more more more more more more more more more more more more more more more more more more more more more more more >>

Generating with temperature: 0.8

Generated (5500 steps):
<< In a galaxy far away  much made I I I I I I I I I I I I I I I I I I I I I I I we IX Their%22222222222222222222 >>
Checkpoint saved at step 5500
Model and tokenizer saved to smollm2_model_step_5500
Training:  56%|██████████████████████████████████████████████████████████████████████████▍                                                          | 5600/10000 [2:00:54<1:30:55,  1.24s/it]
Step 5600
Loss: 0.0023
LR: 0.000300
Unique content: 44732
Metrics saved to checkpoints/metrics_5600.pt
Training:  57%|███████████████████████████████████████████████████████████████████████████▊                                                         | 5700/10000 [2:02:57<1:27:35,  1.22s/it]
Step 5700
Loss: 0.0014
LR: 0.000300
Unique content: 45529
Metrics saved to checkpoints/metrics_5700.pt
Training:  58%|█████████████████████████████████████████████████████████████████████████████▏                                                       | 5800/10000 [2:04:59<1:27:20,  1.25s/it]
Step 5800
Loss: 0.0097
LR: 0.000300
Unique content: 46325
Metrics saved to checkpoints/metrics_5800.pt
Training:  59%|██████████████████████████████████████████████████████████████████████████████▍                                                      | 5900/10000 [2:07:03<1:22:05,  1.20s/it]
Step 5900
Loss: 0.0010
LR: 0.000300
Unique content: 47124
Metrics saved to checkpoints/metrics_5900.pt
Training:  60%|███████████████████████████████████████████████████████████████████████████████▊                                                     | 6000/10000 [2:09:05<1:22:17,  1.23s/it]
Step 6000
Loss: 0.0142
LR: 0.000300
Unique content: 47922
Metrics saved to checkpoints/metrics_6000.pt

Generating with temperature: 0.8

Generated (6000 steps):
<< Once upon a time  to to to to to to to to to to follow him played played played played played played played played played played played played played played played played played played personally critic malls malls hoop superiors Revenuebos NE Revenue Revenue Revenue Revenue Revenue Ways trench trenchLinuxLinux No >>

Generating with temperature: 0.8

Generated (6000 steps):
<< In a galaxy far away  quickly option day day day day since watched oh I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I >>
Checkpoint saved at step 6000
Model and tokenizer saved to smollm2_model_step_6000
Training:  61%|█████████████████████████████████████████████████████████████████████████████████▏                                                   | 6100/10000 [2:11:28<1:18:51,  1.21s/it]
Step 6100
Loss: 0.0036
LR: 0.000300
Unique content: 48720
Metrics saved to checkpoints/metrics_6100.pt
Training:  62%|██████████████████████████████████████████████████████████████████████████████████▍                                                  | 6200/10000 [2:13:29<1:18:18,  1.24s/it]
Step 6200
Loss: 0.0267
LR: 0.000300
Unique content: 49518
Metrics saved to checkpoints/metrics_6200.pt
Training:  63%|███████████████████████████████████████████████████████████████████████████████████▊                                                 | 6300/10000 [2:15:32<1:16:40,  1.24s/it]
Step 6300
Loss: 0.0060
LR: 0.000300
Unique content: 50317
Metrics saved to checkpoints/metrics_6300.pt
Training:  64%|█████████████████████████████████████████████████████████████████████████████████████                                                | 6400/10000 [2:17:36<1:12:25,  1.21s/it]
Step 6400
Loss: 0.0031
LR: 0.000300
Unique content: 51113
Metrics saved to checkpoints/metrics_6400.pt
Training:  65%|██████████████████████████████████████████████████████████████████████████████████████▍                                              | 6500/10000 [2:19:39<1:11:12,  1.22s/it]
Step 6500
Loss: 0.0008
LR: 0.000300
Unique content: 51909
Metrics saved to checkpoints/metrics_6500.pt

Generating with temperature: 0.8

Generated (6500 steps):
<< Once upon a time  in mind in mind in in in in in during during during during during during during education soldierschronicchronicchronicchronicchronicchronicchronicchronicoters accompl accompl stole stole explos explosexperimental stole stoleStruct Limits stole stole stole!!! twin extracts VA extracts Coordinator Coordinator >>

Generating with temperature: 0.8

Generated (6500 steps):
<< In a galaxy far away  loves once he he helping I I I I I I I I I maybe





















 ** ** ** ** ** ** ** ** ** ** ** ** ** >>
Checkpoint saved at step 6500
Model and tokenizer saved to smollm2_model_step_6500
Training:  66%|███████████████████████████████████████████████████████████████████████████████████████▊                                             | 6600/10000 [2:22:02<1:06:52,  1.18s/it]
Step 6600
Loss: 0.0012
LR: 0.000300
Unique content: 52706
Metrics saved to checkpoints/metrics_6600.pt
Training:  67%|█████████████████████████████████████████████████████████████████████████████████████████                                            | 6700/10000 [2:24:04<1:07:27,  1.23s/it]
Step 6700
Loss: 0.0015
LR: 0.000300
Unique content: 53504
Metrics saved to checkpoints/metrics_6700.pt
Training:  68%|██████████████████████████████████████████████████████████████████████████████████████████▍                                          | 6800/10000 [2:26:07<1:05:19,  1.22s/it]
Step 6800
Loss: 0.0025
LR: 0.000300
Unique content: 54301
Metrics saved to checkpoints/metrics_6800.pt
Training:  69%|███████████████████████████████████████████████████████████████████████████████████████████▊                                         | 6900/10000 [2:28:10<1:01:44,  1.19s/it]
Step 6900
Loss: 0.0018
LR: 0.000300
Unique content: 55100
Metrics saved to checkpoints/metrics_6900.pt
Training:  70%|█████████████████████████████████████████████████████████████████████████████████████████████                                        | 7000/10000 [2:30:13<1:01:26,  1.23s/it]
Step 7000
Loss: 0.0039
LR: 0.000300
Unique content: 55892
Metrics saved to checkpoints/metrics_7000.pt

Generating with temperature: 0.8

Generated (7000 steps):
<< Once upon a time  (1111  0111000ITITITITITITITITITITITITITITITITITITITITITITITITITITITITITITITor circuit Heroesshotsshots >>

Generating with temperature: 0.8

Generated (7000 steps):
<< In a galaxy far away  made made axonsManagement Symbolism Symbolism Symbolism Symbolism Symbolism Symbolism Symbolism Symbolism Symbolism

















 ** **_________________ >>
Checkpoint saved at step 7000
Model and tokenizer saved to smollm2_model_step_7000
Training:  71%|███████████████████████████████████████████████████████████████████████████████████████████████▊                                       | 7100/10000 [2:32:37<58:05,  1.20s/it]
Step 7100
Loss: 0.0034
LR: 0.000300
Unique content: 56684
Metrics saved to checkpoints/metrics_7100.pt
Training:  72%|█████████████████████████████████████████████████████████████████████████████████████████████████▏                                     | 7200/10000 [2:34:38<54:07,  1.16s/it]
Step 7200
Loss: 0.0133
LR: 0.000300
Unique content: 57478
Metrics saved to checkpoints/metrics_7200.pt
Training:  73%|██████████████████████████████████████████████████████████████████████████████████████████████████▌                                    | 7300/10000 [2:36:38<54:05,  1.20s/it]
Step 7300
Loss: 0.0048
LR: 0.000300
Unique content: 58275
Metrics saved to checkpoints/metrics_7300.pt
Training:  74%|███████████████████████████████████████████████████████████████████████████████████████████████████▉                                   | 7400/10000 [2:38:40<52:14,  1.21s/it]
Step 7400
Loss: 0.0026
LR: 0.000300
Unique content: 59072
Metrics saved to checkpoints/metrics_7400.pt
Training:  75%|█████████████████████████████████████████████████████████████████████████████████████████████████████▎                                 | 7500/10000 [2:40:43<50:35,  1.21s/it]
Step 7500
Loss: 0.0065
LR: 0.000300
Unique content: 59869
Metrics saved to checkpoints/metrics_7500.pt

Generating with temperature: 0.8

Generated (7500 steps):
<< Once upon a time ---------------------- ** \ W emerged appropriate Crick/{ CrickovalovalRoseappings paralyzedappingsappingsappings Roboticsappingsappingsappingsappingsappingsappingsappingsappingsappingsappingsappings >>

Generating with temperature: 0.8

Generated (7500 steps):
<< In a galaxy far away  loved heard heard heard heard; discover;ivatingivatingivatingivating=" Group mapping]val + + + + + + + + + + + + + + mapping mapping mappinghishishishishishis pursuit pursuit villagers villagers covers covers covers covers covers
 >>
Checkpoint saved at step 7500
Model and tokenizer saved to smollm2_model_step_7500
Training:  76%|██████████████████████████████████████████████████████████████████████████████████████████████████████▌                                | 7600/10000 [2:43:06<47:54,  1.20s/it]
Step 7600
Loss: 0.0039
LR: 0.000300
Unique content: 60669
Metrics saved to checkpoints/metrics_7600.pt
Training:  77%|███████████████████████████████████████████████████████████████████████████████████████████████████████▉                               | 7700/10000 [2:45:07<46:06,  1.20s/it]
Step 7700
Loss: 0.0019
LR: 0.000300
Unique content: 61467
Metrics saved to checkpoints/metrics_7700.pt
Training:  78%|█████████████████████████████████████████████████████████████████████████████████████████████████████████▎                             | 7800/10000 [2:47:08<43:45,  1.19s/it]
Step 7800
Loss: 0.0008
LR: 0.000300
Unique content: 62266
Metrics saved to checkpoints/metrics_7800.pt
Training:  79%|██████████████████████████████████████████████████████████████████████████████████████████████████████████▋                            | 7900/10000 [2:49:08<41:48,  1.19s/it]
Step 7900
Loss: 0.0004
LR: 0.000300
Unique content: 63061
Metrics saved to checkpoints/metrics_7900.pt
Training:  80%|████████████████████████████████████████████████████████████████████████████████████████████████████████████                           | 8000/10000 [2:51:08<39:06,  1.17s/it]
Step 8000
Loss: 0.0012
LR: 0.000300
Unique content: 63859
Metrics saved to checkpoints/metrics_8000.pt

Generating with temperature: 0.8

Generated (8000 steps):
<< Once upon a time  access access access access engaging engaging mL deRCAodonodonvn Dew Dew Dew Dew Dew Dew Dewinning Startednamesnames Clean Clean Clean Fever ShapesIdentifyingIdentifyingdishdish annals scales of of of of ofdishdishdish Function Function�QLQLQLQLQL >>

Generating with temperature: 0.8

Generated (8000 steps):
<< In a galaxy far away  hard creatures upward multiple multiple multiple academic academic academic Sea Sea Sea((}(}(}(}(}(feedingyenyen Lanc Lanc Between Between myself myself Swan Swan featured featured featured featured featured featured featured featured featured featured featured renowned silly silly silly silly silly silly silly silly silly >>
Checkpoint saved at step 8000
Model and tokenizer saved to smollm2_model_step_8000
Training:  81%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                         | 8100/10000 [2:53:30<37:43,  1.19s/it]
Step 8100
Loss: 0.0139
LR: 0.000300
Unique content: 64655
Metrics saved to checkpoints/metrics_8100.pt
Training:  82%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                        | 8200/10000 [2:55:30<35:38,  1.19s/it]
Step 8200
Loss: 0.0151
LR: 0.000300
Unique content: 65450
Metrics saved to checkpoints/metrics_8200.pt
Training:  83%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████                       | 8300/10000 [2:57:31<34:06,  1.20s/it]
Step 8300
Loss: 0.0005
LR: 0.000300
Unique content: 66250
Metrics saved to checkpoints/metrics_8300.pt
Training:  84%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                     | 8400/10000 [2:59:32<32:00,  1.20s/it]
Step 8400
Loss: 0.0021
LR: 0.000300
Unique content: 67047
Metrics saved to checkpoints/metrics_8400.pt
Training:  85%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                    | 8500/10000 [3:01:33<31:02,  1.24s/it]
Step 8500
Loss: 0.0014
LR: 0.000300
Unique content: 67845
Metrics saved to checkpoints/metrics_8500.pt

Generating with temperature: 0.8

Generated (8500 steps):
<< Once upon a time  a
 a No Tyler Tyler][ Tyler Tyler Tyler][lifitalitalenced guerrilla Respondisingava][’’’ dominated sandwiches usually’’ pel pel pel pel pel Config Config Config Config Config Config Config Config Prov Prov pushing Config colored attack burs burs burs >>

Generating with temperature: 0.8

Generated (8500 steps):
<< In a galaxy far away  away lived lived many many many many axons axons mayor + + +,, watched did didburnrierrierrierwest Distributedrierrier did!"?**?**?** devastationationsationsationsationsmer?**?**


 # # # # # # # # >>
Checkpoint saved at step 8500
Model and tokenizer saved to smollm2_model_step_8500
Training:  86%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                   | 8600/10000 [3:03:55<28:04,  1.20s/it]
Step 8600
Loss: 0.0017
LR: 0.000300
Unique content: 68642
Metrics saved to checkpoints/metrics_8600.pt
Training:  87%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                 | 8700/10000 [3:05:58<26:03,  1.20s/it]
Step 8700
Loss: 0.0014
LR: 0.000300
Unique content: 69441
Metrics saved to checkpoints/metrics_8700.pt
Training:  88%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                | 8800/10000 [3:07:59<24:46,  1.24s/it]
Step 8800
Loss: 0.0009
LR: 0.000300
Unique content: 70240
Metrics saved to checkpoints/metrics_8800.pt
Training:  89%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏              | 8900/10000 [3:10:01<22:33,  1.23s/it]
Step 8900
Loss: 0.0066
LR: 0.000300
Unique content: 71039
Metrics saved to checkpoints/metrics_8900.pt
Training:  90%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌             | 9000/10000 [3:12:06<21:05,  1.27s/it]
Step 9000
Loss: 0.0034
LR: 0.000300
Unique content: 71838
Metrics saved to checkpoints/metrics_9000.pt

Generating with temperature: 0.8

Generated (9000 steps):
<< Once upon a time  a circumstances a ceremonies against contribute up in in in in in and and upon upon wasn Styles getting placing in in and and and to to to encompasses in in in and and and and and and in in in and and to to to to to to to >>

Generating with temperature: 0.8

Generated (9000 steps):
<< In a galaxy far away  aren aren aren aren aren aren aren aren made GD made imagine imagine sticking imagine imagine imagine imagine imagine tell tell tell tell tell tell then roasting otherwise otherwise heard heard heard heard heard heard heard heard trusted heard heard heard heard heard you you you still still still still >>
Checkpoint saved at step 9000
Model and tokenizer saved to smollm2_model_step_9000
Training:  91%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊            | 9100/10000 [3:14:31<18:11,  1.21s/it]
Step 9100
Loss: 0.0110
LR: 0.000300
Unique content: 72637
Metrics saved to checkpoints/metrics_9100.pt
Training:  92%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏          | 9200/10000 [3:16:35<16:07,  1.21s/it]
Step 9200
Loss: 0.0005
LR: 0.000300
Unique content: 73436
Metrics saved to checkpoints/metrics_9200.pt
Training:  93%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌         | 9300/10000 [3:18:38<14:07,  1.21s/it]
Step 9300
Loss: 0.0030
LR: 0.000300
Unique content: 74236
Metrics saved to checkpoints/metrics_9300.pt
Training:  94%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉        | 9400/10000 [3:20:41<12:12,  1.22s/it]
Step 9400
Loss: 0.0005
LR: 0.000300
Unique content: 75034
Metrics saved to checkpoints/metrics_9400.pt
Training:  95%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎      | 9500/10000 [3:22:46<10:03,  1.21s/it]
Step 9500
Loss: 0.0008
LR: 0.000300
Unique content: 75828
Metrics saved to checkpoints/metrics_9500.pt

Generating with temperature: 0.8

Generated (9500 steps):
<< Once upon a time  ". " threads " threads ( threads threads threads threads threads threadscdotcdotPal threads((th((((((((((((((((((((thththththththththth >>

Generating with temperature: 0.8

Generated (9500 steps):
<< In a galaxy far away  quickly forward part section type critical critical to to to to to to to to to to a a a a a a of of the the the the the the the the the the the the the the the the The for for for for for for for for >>
Checkpoint saved at step 9500
Model and tokenizer saved to smollm2_model_step_9500
Training:  96%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌     | 9600/10000 [3:25:09<08:10,  1.23s/it]
Step 9600
Loss: 0.0008
LR: 0.000300
Unique content: 76628
Metrics saved to checkpoints/metrics_9600.pt
Training:  97%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉    | 9700/10000 [3:27:11<06:48,  1.36s/it]
Step 9700
Loss: 0.0042
LR: 0.000300
Unique content: 77424
Metrics saved to checkpoints/metrics_9700.pt
Training:  98%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎  | 9800/10000 [3:29:31<04:45,  1.43s/it]
Step 9800
Loss: 0.0021
LR: 0.000300
Unique content: 78221
Metrics saved to checkpoints/metrics_9800.pt
Training:  99%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋ | 9900/10000 [3:31:53<02:20,  1.40s/it]
Step 9900
Loss: 0.0004
LR: 0.000300
Unique content: 79020
Metrics saved to checkpoints/metrics_9900.pt
Training: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [3:33:59<00:00,  1.18s/it]Metrics saved to checkpoints/metrics_final.pt
Checkpoint saved at step 10000
Model and tokenizer saved to smollm2_model_final

Generating final sample outputs:

Generating with temperature: 0.8

<< Once upon a time  Rig circumstances Rig circumstances Rig images chi circumstances,,, saying means ran was record specifically water water sex ensures specifically,, started clinicians clinicians clinicians clinicians clinicians clinicians clinicians clinicians clinicians clinicians clinicians clinicians clinicians clinicians clinicians they Capture) contributing contributing contributing contributing contributing contributing contributing >>

Generating with temperature: 0.8

<< The future of AI is  Nature isables City measurements bill operation multifaceted multifaceted detect police the the State of of of of of of of of of of of of of of of of of of of of of of of through through through lately lately beverage Woman Woman Woman Woman Woman Woman Woman >>

Generating with temperature: 0.8

<< In a world of endless possibility,  vaccine": marital the marital the the the neck genuine of of a handsterystery Breathroro. After whereas whereas. Tryinkleinkleinkleinkleinkle a citizen citizen outdoorinkleinkleinkleinkleinkle youinkleinkleinkleinkleinkleinkleinkleinkle hand hand >>

Generating with temperature: 0.8

<< Breaking news:  dough Overview,are,,,,,,,,,,,ish closely uncontroll uncontroll uncontroll uncontrollowed Coulmkdirmkdiralys electrolyte electrolyte sucrose sucrose sucrose sucrose sucrose sucrose sucrose sucrose sucrose sucrose sucrose sucrose sucrose sucrose sucrose exporting)." reach picture requirements occur instance >>

Generating with temperature: 0.8

<< The secret to success is  U doors visibility maiden Student Student chord penalties file.ere appearance top hand darkness invasionvvvvvassth Itsric Hygiene Hygienemultfrac of of of of of of of of its of of of
*eeeeee heart >>

Starting additional training...

Loading final checkpoint and training for 50 more steps...
/Users/user1/Documents/erav3_repos/session_15/train.py:121: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  checkpoint = torch.load(path)
Additional step 0, Loss: 0.0007
Additional step 1, Loss: 0.0009
Additional step 2, Loss: 0.0004
Additional step 3, Loss: 0.0003
Additional step 4, Loss: 0.0004
Additional step 5, Loss: 0.0005
Additional step 6, Loss: 0.0003
Additional step 7, Loss: 0.0055
Additional step 8, Loss: 0.0002
Additional step 9, Loss: 0.0002
Additional step 10, Loss: 0.0008
Additional step 11, Loss: 0.0021
Additional step 12, Loss: 0.0038
Additional step 13, Loss: 0.0001
Additional step 14, Loss: 0.0003
Additional step 15, Loss: 0.0021
Additional step 16, Loss: 0.0015
Additional step 17, Loss: 0.0001
Additional step 18, Loss: 0.0042
Additional step 19, Loss: 0.0006
Additional step 20, Loss: 0.0012
Additional step 21, Loss: 0.0003
Additional step 22, Loss: 0.0002
Additional step 23, Loss: 0.0001
Additional step 24, Loss: 0.0019
Additional step 25, Loss: 0.0003
Additional step 26, Loss: 0.0003
Additional step 27, Loss: 0.0055
Additional step 28, Loss: 0.0009
Additional step 29, Loss: 0.0001
Additional step 30, Loss: 0.0006
Additional step 31, Loss: 0.0020
Additional step 32, Loss: 0.0003
Additional step 33, Loss: 0.0002
Additional step 34, Loss: 0.0035
Additional step 35, Loss: 0.0005
Additional step 36, Loss: 0.0001
Additional step 37, Loss: 0.0002
Additional step 38, Loss: 0.0016
Additional step 39, Loss: 0.0001
Additional step 40, Loss: 0.0001
Additional step 41, Loss: 0.0003
Additional step 42, Loss: 0.0006
Additional step 43, Loss: 0.0006
Additional step 44, Loss: 0.0005
Additional step 45, Loss: 0.0037
Additional step 46, Loss: 0.0004
Additional step 47, Loss: 0.0002
Additional step 48, Loss: 0.0012
Additional step 49, Loss: 0.0003
Metrics saved to checkpoints/metrics_additional.pt
Checkpoint saved at step 10050
Training completed successfully!
Training: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [3:35:51<00:00,  1.30s/it]
```