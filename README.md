# pun-detection
This is for AI final project report use.
It use BLOOM-560 AI as base model.
The model and the datasets are all fetched through hugging face API. Training and testing by using transformer module.
I adjust the "learning_rate", "num_train_epochs", and "weight_decay" these 3 training parameters.
To try which may result the best accuracy.

# report1.py
learning_rate=2e-5, num_train_epochs=6, weight_decay=0.01.

# report2.py
learning_rate=1.5e-5, num_train_epochs=6, weight_decay=0.01.

# report3.py
learning_rate=1.3e-5, num_train_epochs=6, weight_decay=0.007.

# report4.py
learning_rate=1.3e-5, num_train_epochs=5, weight_decay=0.015.

# report5.py
learning_rate=1.7e-5, num_train_epochs=4, weight_decay=0.013.

# report6.py
learning_rate=2.3e-5, num_train_epochs=4, weight_decay=0.013.
