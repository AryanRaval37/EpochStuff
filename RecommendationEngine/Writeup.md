# Write Up

Disclaimer: I had very little time to do this because of my exams and travel plan. There was more AI used in the making of this than I would have liked😢
All of the logic is still written by me, however I used quite a lot of AI for the documentation, refactoring into different files, model saving loading and some bug fixing.

Had a look at this paper: https://arxiv.org/pdf/1708.05031

- First implemented Matrix Factorisation:
    - new random negative sampling is quite important for the model to learn (first mistake fixed - hitK went from 19 to 35)
    - found other bugs in the model and the evalulation metric (something to do with uniqueness of the datapoints) (hitk already up to 50)
    - found that the num negatives the model sees while training is quite important, they quite dramatically affect hitk, when set to 100 (same as the evaluation) hitk went to about 60. (increasing negative sampling even more from 100 to 150 or something while training would probably help actually)
    - also added a bias term that was not there initially
    
- Then went to implement MLP:
    - this was way more trivial
    - most of the code was standard - reading that NeuroMF research paper, learnt the trick of providing both the embeddings and their products for some reason
    - MLP always gives a slightly lower hitk than MF probably because its a way more general model not build for specifically handling the particular task of dot products between user - item interactions

- Then went to implement NeuMF:
    - Because of lack of time, I just read through the paper to understand the architecture explain claude exactly what I wanted and let it do the implementation.
    - Debugging was severely more time consuming than I expected (as is with almost any AI generated code😔) 
    - In any case combing the two models in this way does consistently give a better hitk than either of the models individually by about 4-5 points more.
    - Like mentioned in the paper its very good to have pretraining for the two models, also necessary to keep learning rates much lower and weight decays much higher whlie fine tuning the NeuMF otherwise overfitting is very common reducing the performance by 4-5 points compared to the individual models.


- Instead of NeuMF idea of having an MF, MLP and just taking a linear combination, I would like to have attention heads that learn the combination themselves. I will implement this once I'm back. 
- I would also like to use the BPR loss function instead of BCEWithLogitsLoss since it is more suited for ranking tasks (i think... Haven't looked into it properly)
There are many more things that are scattered through the code in comments in all the tasks which are like tangents that I would like to go on once the initial implementation is complete. Also would like to a proper final refactoring this project to make it overall more usuable.








