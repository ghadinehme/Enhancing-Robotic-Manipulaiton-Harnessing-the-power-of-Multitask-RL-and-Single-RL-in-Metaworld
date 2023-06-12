LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so python3 train.py use_discrim=True rl_pretraining=True q_weights=True online_steps=200000 env_name=sawyer_pick_place


LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so python env_viewer.py