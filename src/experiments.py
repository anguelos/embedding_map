#!/usr/bin/env python
import sys
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
import seaborn
import map
from scipy.spatial.distance import cdist
from scipy.ndimage.filters import gaussian_filter

def exploit_ambiguity(nb_samples=1000,embbeding_size=100,nb_classes=10,nb_random_repeat=30,report=False):
    zero_features=np.zeros([nb_samples,embbeding_size])
    unsorted_labels = np.arange(nb_samples)%nb_classes
    sorted_labels = np.sort(unsorted_labels)
    shuffled_labels=sorted_labels.copy()
    np.random.shuffle(shuffled_labels)
    zero_map=np.zeros([1,5])
    zero_map[0, 0] = map.get_map(zero_features,unsorted_labels,mode="unspecified")
    zero_map[0, 1] = map.get_map(zero_features, sorted_labels, mode="unspecified")
    zero_map[0, 2] = map.get_map(zero_features, shuffled_labels,
                                 mode="unspecified")
    zero_map[0, 3] = map.get_map(zero_features, sorted_labels, mode="pessimistic")
    zero_map[0, 4] = map.get_map(zero_features, sorted_labels, mode="optimistic")

    rnd_map=np.zeros([nb_random_repeat,3])
    for k in range(nb_random_repeat):
        rnd_features = np.random.rand(*zero_features.shape)
        rnd_map[k,0] =  map.get_map(rnd_features,unsorted_labels,mode="unspecified")
        rnd_map[k, 1] = map.get_map(rnd_features, sorted_labels, mode="unspecified")
        rnd_map[k, 2] = map.get_map(rnd_features, shuffled_labels, mode="unspecified")

    if report:
        plt.clf()
        plt.plot(unsorted_labels)
        plt.plot(sorted_labels)
        plt.legend(["mAP 10.31%", "mAP 18.68%"])
        plt.title("All 0 embedding labeling")
        plt.xlabel("Sample")
        plt.ylabel("Label")
        plt.savefig("all_zero_map.pdf", bbox_inches='tight')
        rnd_map_sorted=np.sort(rnd_map[:,:2],axis=1)
        print "Zero embbeding exploit:\nRandom Embedings:"
        print "\tOver %d repetitions: min:%3.2f max:%3.2f"%(nb_random_repeat,rnd_map_sorted.min()*100,rnd_map_sorted.max()*100)
        least_range_idx=np.argmin(rnd_map_sorted[:,1]-rnd_map_sorted[:,0])
        greatest_range_idx =np.argmax(rnd_map_sorted[:,1]-rnd_map_sorted[:,0])
        print "\tSmallest range in %d repetitions: min:%3.2f max:%3.2f"%(nb_random_repeat,rnd_map_sorted[least_range_idx,0]*100,rnd_map_sorted[least_range_idx,1]*100)
        print "\tGreatest range in %d repetitions: min:%3.2f max:%3.2f" % (nb_random_repeat, rnd_map_sorted[greatest_range_idx, 0] * 100,rnd_map_sorted[greatest_range_idx,1]*100)
        print "\tmean mAP over %d repetions %3.2f with a deviation of %3.4f" %(nb_random_repeat,(rnd_map[:,:2]*100).mean(),(rnd_map[:,:2]*100).std())
        print "Zero emdeddings:\n\tmin (shufled labels):%3.2f, max(sorted_labels):%3.2f, expectation(shuffled):%3.2f" % (zero_map[0,0]*100,zero_map[0,1]*100,zero_map[0,2]*100)
        print "\tmin(pessimist):%3.2f, max(optimist):%3.2f"%(zero_map[0,3]*100,zero_map[0,4]*100)


def load_phocnet_data():
    phocnet_data_path = "./data/phocnet.npz"
    phoc_data = np.load(phocnet_data_path)
    phoc_features = phoc_data["phocnet_features"]
    phoc_labels = phoc_data["phocnet_labels"].reshape(-1)
    return phoc_features, phoc_labels


def quantize(data,quanta):
    data_min =data.min()
    data_max = data.max()
    data=(data-data_min)/(data_max-data_min)
    res= np.floor(data*quanta)/quanta
    return res*(data_max-data_min)+data_min

def get_map_bounds(query_fetaures,query_labels,retrieval_features,retrieval_labels,metric="cosine",e=1e-20):
    dist_mat = cdist(query_fetaures, retrieval_features, metric)#*100).astype("int32")/100
    correct=(retrieval_labels[None,:]==query_labels[:,None]).astype("float64")
    #print dist_mat.min(axis=1).max()


    #print correct.shape,correct.sum(axis=1).min()

    optimist = dist_mat #- correct * e
    pessimist = dist_mat #+ correct * e
    results =[]
    for dm in [optimist,pessimist]:
        indexes = np.argsort(dm,axis=1)
        correct_retrivals=correct[np.arange(dist_mat.shape[0])[:,np.newaxis],indexes]
        plt.imshow(correct_retrivals);
        plt.show()
        if (query_fetaures == retrieval_features).all():
            assert query_labels is retrieval_labels
            correct_retrivals=correct_retrivals[:,1:]

        correct_at = np.cumsum(correct_retrivals,axis=1)

        precision_at = correct_at/np.cumsum(np.ones_like(correct_retrivals),axis=1)
        #recall_at = correct_at / (e+correct_retrivals.sum(axis=1)[:, None])
        #padded_recall_at = np.concatenate(
        #    (np.zeros([recall_at.shape[0], 1]), recall_at), axis=1)
        #recall_changes_at = padded_recall_at[:, 1:] > padded_recall_at[:, :-1]

        #average_precison = (precision_at * recall_changes_at).sum(
        #    axis=1) / recall_changes_at.sum(axis=1)
        #print '\nCor',correct_at.sum()
        #print 'P@',precision_at.sum()
        #print 'R@', recall_at[:,-1].min()
        #print 'RD@',recall_changes_at.sum(),recall_changes_at.sum(axis=1).min()
        #print 'AP',average_precison.sum()
        results.append(average_precison.mean())
    return results



def plot_dist_colisions(query_fetaures,query_labels,retrieval_features,retrieval_labels,metric="cosine",e=1e-10,quanta=None,fname=None):
    plt.clf()
    #dist_mat=cdist(query_fetaures,retrieval_features,"euclidean")
    dist_mat = cdist(query_fetaures, retrieval_features, metric)
    if quanta:
        dist_mat=quantize(dist_mat,100000000)
        #print "Colision probabillity:",1/(quanta/e)
    #dist_mat = map.cosine_distance(query_fetaures, retrieval_features.T)
    idx=np.argsort(dist_mat,axis=1)

    sorted_dist=dist_mat[np.arange(dist_mat.shape[0])[:,np.newaxis],idx]
    correct=(retrieval_labels[idx[:,1:]]==query_labels[:,None]).astype("float")

    #non_singleton=np.nonzero(correct.sum(axis=1)>1)[0]
    #sorted_dist=sorted_dist[non_singleton,:]
    #correct=correct[non_singleton,:]

    wrong=1-correct
    collision=((sorted_dist[:,1:]-sorted_dist[:,:-1])<e).astype("float")
    #collision=dilate_colisions(collision,3)
    visible_colision=collision*correct-collision*wrong
    plt.imshow(visible_colision,cmap="bwr",vmin="-1", vmax="1")#)"RdYlGn")
    #plt.imshow(correct);
    cbar=plt.colorbar(ticks=[-1,0,1])
    cbar.ax.set_yticklabels(["Irelevant\nColision", "No\nColision", "Relevant\nColision"])
    if fname is None:
        plt.show()
    else:
        words=fname.split("/")[-1].split(".")[0].split("_")
        words=[w[0].upper()+w[1:] for w in words if w]
        title = " ".join(words)
        plt.title(title)
        plt.savefig(fname, bbox_inches='tight')

def plot_phocnet_colisions():
    phoc_features, phoc_labels = load_phocnet_data()

    #cherry picking an indicative subset of the GW test-set for decent visualition
    indexes=np.arange(650,phoc_labels.shape[0],dtype="int32")
    plot_dist_colisions(phoc_features[indexes],phoc_labels[indexes],phoc_features[indexes],phoc_labels[indexes],metric="cosine",fname="_GW_PHOCNET_indicative_retrievals_cosine.pdf")
    plot_dist_colisions(phoc_features[indexes],phoc_labels[indexes],phoc_features[indexes],phoc_labels[indexes],metric="euclidean",fname="_GW_PHOCNET_indicative_retrievals_euclidean.pdf")
    plot_dist_colisions(phoc_features[indexes],phoc_labels[indexes],phoc_features[indexes],phoc_labels[indexes],metric="cityblock",fname="_GW_PHOCNET_indicative_retrievals_cityblock.pdf")

def plot_whitenoise_colisions():
    e=1e-5
    phoc_features, phoc_labels = load_phocnet_data()
    rnd_features = np.random.rand(*phoc_features.shape)
    rnd_labels = phoc_labels

    #cherry picking an indicative subset of the GW test-set for decent visualition
    indexes=np.arange(900,rnd_labels.shape[0],dtype="int32")
    plot_dist_colisions(rnd_features[indexes],rnd_labels[indexes],rnd_features[indexes],rnd_labels[indexes],metric="cosine",fname="_white_noise_embeddings_indicative_retrievals_cosine.pdf",e=e)
    plot_dist_colisions(rnd_features[indexes],rnd_labels[indexes],rnd_features[indexes],rnd_labels[indexes],metric="euclidean",fname="_white_noise_embeddings_indicative_retrievals_euclidean.pdf",e=e)
    plot_dist_colisions(rnd_features[indexes],rnd_labels[indexes],rnd_features[indexes],rnd_labels[indexes],metric="cityblock",fname="_white_noise_embeddings_indicative_retrievals_cityblock.pdf",e=e)

def plot_phocnet_e_mask():
    phocnet_data_path = "./data/phocnet.npz"
    phoc_data = np.load(phocnet_data_path)
    phoc_features = phoc_data["phocnet_features"]
    phoc_labels = phoc_data["phocnet_labels"].reshape(-1)

    #cherry picking an indicative subset of the GW test-set for decent visualition
    indexes=np.arange(900,phoc_labels.shape[0],dtype="int32")
    q_features=phoc_features[indexes]
    q_labels=phoc_labels[indexes]
    db_features=phoc_features[indexes]
    db_labels=phoc_labels[indexes]
    dist_mat = cdist(q_features, db_features, "cosine")
    correct=(db_labels[None,:]==q_labels[:,None]).astype("float")

    plt.clf()
    plt.imshow(1-correct)
    cbar=plt.colorbar(ticks=[0,1])
    cbar.ax.set_yticklabels(["0","e"])
    plt.title("Matrix E")
    plt.savefig("Me.pdf", bbox_inches='tight')

    plt.clf()
    plt.imshow(dist_mat)
    cbar=plt.colorbar()
    #cbar.ax.set_yticklabels(["0","e"])
    plt.title("Matrix D")
    plt.savefig("D.pdf", bbox_inches='tight')

def main(phocnet_data_path):
    np.random.seed(1337)
    exploit_ambiguity(report=True)
    plot_phocnet_colisions()
    plot_phocnet_e_mask()
    plot_whitenoise_colisions()


if __name__=="__main__":
    main(sys.argv[1])
