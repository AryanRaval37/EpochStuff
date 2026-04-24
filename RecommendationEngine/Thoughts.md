# Thoughts on how I would do it even before I knew matric factorisation

### Only positive labels available
- it says make a random samples of negative interaction
- probably something like randomly set 0 to some of the movies that the user didn't interact with
- why not just make a 1447 (number of unique movies) dim vector of each user and set 1 wherever user interacted?
- I think it should be fine because the user actually didn't 'interact' with the movie - its not the rating of liked or not
- If we take a random sample, the assumption feels kind of like the user liked these movies and randomly set some other movies to 0 meaning the user didn't particularly like
- think about the 942 by 1447 matrix

### Expectations for what the model can learn:
- there is no data on the movie given - comedy, horror, action or anything
- thats kind of the point of the thing to separate the movies into these high dim embedded vectors
- what a reasonable thing (the only thing i can think of) the model can do is if user P watches movies A B and C and user Q watches movies B C and D then recommend D to P and A to Q
- this is just the distance between the user vectors. 
- for user X find the closest neighbor and make X watch the movie thats in that list
- very KNN type thing

### User vector embeddings
- so instead of using this 1447 dim vector for user, i guess this data could be compressed into a much smaller vector
- train the embedding model based on this interaction data to tune these vectors and then take dot product
- the process should look like relaxing the vectors close together and furthur away from each other
- in the end just take dot product implies this dot product should play a key role in the loss function
- I dont see why this should be any better then my original idea
- I guess this method encapsulates interaction of multiple users, If the person Q is a wastefellow and likes B and C and D is a completely unrelated movie then it will be caught and pushed away by comparing multiple other users.
- what algorithm could i use to implement this?

Time to take a look at matrix factorisation
- before that question: does these also need to be a similar embeddings of movies?? that does seem important, maybe in the original idea if I just find the closest user and sample a random movie that that user watched but not this user is a possibility. However if i manage to embed both user and movies in the same dim space...

Where do MLP's come in?
- for embeddings?? or for the final step of predicting which movie once the user embedding is done?
- in either case how do what do i train the model against??
- its stupid to just give user id and expect the movie out because it will give only the movies the user has interacted with
- the job of entangling data from all the users dimentionality reduction is just asking for there for be transformer models

- At the end of the day I need to know how to embed vectors at all and read matrix factorisation