# Anti-suicide-chatbot

The algo is using a neural network of 16 neurons to predict the right output based on the input in the intents.json file witch contains some 
responses that are ment to help people with mental health issues like addiction, suicidal problems, abuse and breakup problems.

In the first part of the code, we are trainig the model on the intents.json file, then we fit and save the model using :

	model.fit(training, output, n_epoch=2000, batch_size=8, show_metric=True)
	model.save("model.tflearn")


After that we convert the input into a bag of words of type [1, 0, 1, 0, 0, 1, ... ] based on the frequency of the words in the input using: 
	
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)

In the final part of the code we have the chat() function that recives the input from the user and makes the prediction and only returns a value
if the accuracy is > 80%:

def chat():
    print("Start talking with the bot!")

    while True:
        inp = input("Tell me what is on your heart: ")
        if inp.lower() == "quit":
            break
        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = numpy.argmax(results)
        if results[results_index] > 0.8:
            tag = labels[results_index]
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg["responses"]
                    print(random.choice(responses))
        else:
            print("I don't understand. Try Again !")
chat()


The algorithm uses term frequency to determin the best response for the given input. 
 
