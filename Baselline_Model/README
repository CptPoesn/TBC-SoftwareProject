Uploaded files
1. rasa_env.yml
  - environment for Rasa and predictors requirements
2. our_dialogue_manager.py
  - for running model on single input
3. evaluation.py
  - for batch evaluation
4. config.yml
  - replacement of config.yml from rasa initial model
5. training data files
  - switchboard data adapted to rasa yml format
6. evaluation data files
  - switchboard data adapted to rasa yml format

----------------------------------------------------------------------------------------------------------------

Software Requirements
1. Anaconda3
2. Spyder from Anaconda3
3. environment file rasa_env.yml

----------------------------------------------------------------------------------------------------------------

Please run by doing the followings:
A. Set up an environment
  1. In Anaconda Prompt, go to the path of the rasa_env.yml file
  2. Activate the environment with the command ```conda activate rasa_test```
  #fyi: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

B. Train a Rasa model with provided data
  1. Go to the path intended to have your model saved
  2. Get up a new initial Rasa model with command "rasa init"
  3. Replace the initial nlu.yml with training data files (in "data" folder)
  4. Replace the initial config.yml file with the one uploaded here
  5. Run the command `rasa train` in the directory that contains `config.yml` and 

C. Run model on a single sentence as input
  1. Change the path to your trained model's path, i.e. the directory you created in step B.2. (line 266)
  2. Input a sentence for the variable input_msg (line 256)
  3. If desired, change the variable threshold (line 257)
  4. Run the command `python our_dialogue_manager.py`

D. Run evaluation
  1. Adapt the paths for your trained model and the evaluation data corresponding to your training data (lines 8,9)
  2. Feel free to manipulate the hyper-parameters (lines 10, 25-32)
  3. Run `python evaluation.py`
