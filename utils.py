import numpy as np
import torch
import logging
#import losses
import json
from tqdm import tqdm
import torch.nn.functional as F
import math

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)

    return output

def calc_recall_at_k(T, Y, k):
    """
    T : [nb_samples] (target labels)
    Y : [nb_samples x k] (k predicted labels/neighbours)
    """

    s = 0
    for t,y in zip(T,Y):
        if t in torch.Tensor(y).long()[:k]:
            s += 1
    return s / (1. * len(T))

def calc_map_at_k0(T, Y, k):
    """
    T : [nb_samples] (target labels)
    Y : [nb_samples x k] (k predicted labels/neighbours)
    """

    s = 0
    i = 0
    ap =0
    for t,y in zip(T,Y):
        i+=1
        if t in torch.Tensor(y).long()[:k]:
            s += 1
            ap+=s/i
    return ap/i

def calc_map_at_k(T, Y, k):
    """
    T : [nb_samples] (target labels)
    Y : [nb_samples x k] (k predicted labels/neighbours)
    """

    map = 0
    for t,y in zip(T,Y):
        s = 0
        count = 0
        ap = 0
        for i in range(k):
            count = count+1
            if t == y[i]:
                s+=1;
                ap+=s/count;
        map+=ap/k;
    # print(len(Y))
    return map/len(Y)

def predict_batchwise(model, dataloader):
    device = "cuda"
    model_is_training = model.training
    model.eval()
    
    ds = dataloader.dataset
    A = [[] for i in range(len(ds[0]))]
    with torch.no_grad():
        # extract batches (A becomes list of samples)
        for batch in tqdm(dataloader):
            for i, J in enumerate(batch):
                # i = 0: sz_batch * images
                # i = 1: sz_batch * labels
                # i = 2: sz_batch * indices
                if i == 0:
                    # move images to device of model (approximate device)
                    J = model(J.cuda())

                for j in J:
                    A[i].append(j)
    model.train()
    model.train(model_is_training) # revert to previous training state
    
    return [torch.stack(A[i]) for i in range(len(A))]

def proxy_init_calc(model, dataloader):
    nb_classes = dataloader.dataset.nb_classes()
    X, T, *_ = predict_batchwise(model, dataloader)

    proxy_mean = torch.stack([X[T==class_idx].mean(0) for class_idx in range(nb_classes)])

    return proxy_mean

def evaluate_cos(model, dataloader):
    nb_classes = dataloader.dataset.nb_classes()

    # calculate embeddings with model and get targets
    X, T = predict_batchwise(model, dataloader)
    X = l2_norm(X)
    #进行二值化
    #X = torch.sign(X)
    # get predictions by assigning nearest 8 neighbors with cosine
    K = 32
    Y = []
    xs = []
    
    cos_sim = F.linear(X, X)
    Y = T[cos_sim.topk(1 + K)[1][:,1:]]
    Y = Y.float().cpu()
    
    recall = []
    for k in [1, 2, 4, 8, 16, 32]:
        r_at_k = calc_recall_at_k(T, Y, k)
        recall.append(r_at_k)
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))

    return recall

def t2b(num=0):
    res = []
    i = 0
    while(num):
        ys = num % 2
        num = num // 2
        res[i] = ys
        i = i + 1
    return res.reverse()


def evaluate_hamming2(model, dataloader, hash_K, embed_label):
    print('eveluate hamming')
    nb_classes = dataloader.dataset.nb_classes()

    # calculate embeddings with model and get targets
    X, T = predict_batchwise(model, dataloader)
    X = l2_norm(X)
    # 进行二值化
    X = torch.sign(X)

    _, cur_class = torch.max(embed_label,1)
    bClass = t2b(cur_class)
    hashCode = torch.cat((X,bClass),0)
    # get predictions by assigning nearest 8 neighbors with cosine
    K = 80

    cos_sim = F.linear(X, X)
    for i in range(len(X)):
        for j in range(len(X)):
            cos_sim[i][j]=hash_K-hamming_distance(hashCode[i],hashCode[j]);
    Y = T[cos_sim.topk(1 + K)[1][:, 1:]]
    Y = Y.float().cpu()

    aps = []
    map_at_k = calc_map_at_k0(T, Y, 20)
    aps.append(map_at_k)
    print("map@{} : {:.3f}".format(20, 100 * map_at_k))

    return aps

def hamming_distance(x, y):
    # res=0
    res=torch.abs(x-y).sum()/2
    return res

def get_k_hamming_neighbours(nrof_neighbors, enc_test, test_lab, index, test_encodings, test_labs):
    _neighbours = []  # 1(query image) + nrof_neighbours
    distances = []
    print(len(enc_test))
    print(test_labs[0])
    for i in range(len(test_encodings)):
        if index != i:  # exclude the test instance itself from the search set
            # print(test_encodings[i,:])
            dist = hamming_distance(test_encodings[i, :], enc_test)
            print(dist)
            # dist = euclidean_distance(test_encodings[i, :], enc_test)
            #distances = torch.cat([distances,(test_labs[i], dist)],dim=0)
            #dist = torch.tensor.numpy().tolist()
            distances.append(( test_labs[i], dist))

    distances.sort()
    print('distances=',distances)
    _neighbours.append((enc_test, test_lab))
    for j in range(nrof_neighbors):
        _neighbours.append((distances[j][0], distances[j][1]))

    return _neighbours
def get_mAP(neghbours_list, nrof_neighbors):
    test_sample_label = neghbours_list[0][1]
    acc = np.empty((0,)).astype(np.float)
    correct = 1
    for i in range(1, nrof_neighbors+1):
        print(test_sample_label)
        print(neghbours_list[i])
        if test_sample_label == neghbours_list[i][1]:
            precision = (correct / float(i))
            acc = np.append(acc, [precision, ], axis=0)
            correct += 1
    if correct == 1:
        return 0.
    num = np.sum(acc)
    den = correct - 1
    return num/den


def evaluate_hamming(model, dataloader):
    nb_classes = dataloader.dataset.nb_classes()

    # calculate embeddings with model and get targets
    X, T = predict_batchwise(model, dataloader)
    X = l2_norm(X)
    # 进行二值化
    X = torch.sign(X)
    # get predictions by assigning nearest 8 neighbors with cosine
    # K = 20
    total_map = 0
    for i in range(len(X)):
        neighbours = get_k_hamming_neighbours(nrof_neighbors=80, enc_test=X[i], test_lab=T[i], index=i, test_encodings=X, test_labs=T)
        map = get_mAP(neighbours, K)
        total_map += map
    #cos_sim = F.linear(X, X)
    # hamming_dis =
    #Y = T[cos_sim.topk(1 + K)[1][:, 1:]]
    print("map@{} : {:.3f}".format(K, 100 * (total_map / len(X))))

    return 100 * (total_map / len(X))


def evaluate_cos_Inshop(model, query_dataloader, gallery_dataloader):
    nb_classes = query_dataloader.dataset.nb_classes()
    
    # calculate embeddings with model and get targets
    query_X, query_T = predict_batchwise(model, query_dataloader)
    gallery_X, gallery_T = predict_batchwise(model, gallery_dataloader)
    
    query_X = l2_norm(query_X)
    gallery_X = l2_norm(gallery_X)
    
    # get predictions by assigning nearest 8 neighbors with cosine
    K = 50
    Y = []
    xs = []
    
    cos_sim = F.linear(query_X, gallery_X)

    def recall_k(cos_sim, query_T, gallery_T, k):
        m = len(cos_sim)
        match_counter = 0

        for i in range(m):
            pos_sim = cos_sim[i][gallery_T == query_T[i]]
            neg_sim = cos_sim[i][gallery_T != query_T[i]]

            thresh = torch.max(pos_sim).item()

            if torch.sum(neg_sim > thresh) < k:
                match_counter += 1
            
        return match_counter / m
    
    # calculate recall @ 1, 2, 4, 8
    recall = []
    for k in [1, 20, 40, 50]:
        r_at_k = recall_k(cos_sim, query_T, gallery_T, k)
        recall.append(r_at_k)
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))
                
    return recall

def evaluate_cos_SOP(model, dataloader):
    nb_classes = dataloader.dataset.nb_classes()
    
    # calculate embeddings with model and get targets
    X, T = predict_batchwise(model, dataloader)
    X = l2_norm(X)
    
    # get predictions by assigning nearest 8 neighbors with cosine
    K = 1000
    Y = []
    xs = []
    for x in X:
        if len(xs)<10000:
            xs.append(x)
        else:
            xs.append(x)            
            xs = torch.stack(xs,dim=0)
            cos_sim = F.linear(xs,X)
            y = T[cos_sim.topk(1 + K)[1][:,1:]]
            Y.append(y.float().cpu())
            xs = []
            
    # Last Loop
    xs = torch.stack(xs,dim=0)
    cos_sim = F.linear(xs,X)
    y = T[cos_sim.topk(1 + K)[1][:,1:]]
    Y.append(y.float().cpu())
    Y = torch.cat(Y, dim=0)

    # calculate recall @ 1, 2, 4, 8
    recall = []
    for k in [1, 10, 100, 1000]:
        r_at_k = calc_recall_at_k(T, Y, k)
        recall.append(r_at_k)
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))
    return recall
