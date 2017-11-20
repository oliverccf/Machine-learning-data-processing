import pickle

with open('../final_project_dataset.pkl', 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    data = pickle.load(f)

    print("Total person: ",len(data.keys()))
    print("Tatal feature of a person: ",len(data[next(iter(data))]))
    print("Total person of interest : ",
          (sum([int(data[person]['poi'] == 1) for person in data])))
