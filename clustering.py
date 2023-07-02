from clustering_metrics import clustering_metrics
import argparse
from model import Encoder
from utils import load_dataset

class Clustering(torch.nn.Module):
    def __init__(self, labels, n_clusetrs, n_seeds):
        super().__init__()
        self.labels = labels
        self.n_clusters = n_cluster
        self.n_seeds = n_seeds
        self.nmi = []
        self.acc = []
        self.f1 = []

    def forward(self, z)
        for seed in range(self.n_seeds):
            kmeans = KMeans(n_clusters=n_clusters, random_state=seed).fit(z)
            predict_labels = kmeans.predict(z)
            metrics = clustering_metrics(self.labels, predict_labels)
            acc, nmi, f1 = metrics.evaluationClusterModelFromLabel()
            self.acc.append(acc)
            self.nmi.append(nmi)
            self.f1.append(f1)

        return self.acc, self.nmi, self.f1

    def load_model(self, model, data, path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["encoder"])
        model.eval()
        # inference
        z = model(data.x, data.edge_index)
        z = z.detach().cpu().numpy()
        return z

    def load_embed(self, path):
        embed = np.load(path)
        return embed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--dataset", type=str, required=True, default="cora")
    parser.add_argument("--load-model", action="store_true", required=True)
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--n-seeds", type=int, description="number of seeds for clustering", required=true)
    parser.add_argument("--n-model-seeds", type=int, description="number of seeds for model training", required=true)
    parser.add_argument("--normalization", action="store_true")
    parser.add_argument('--pre_norm', action="store_true") # dataset dependent

    args = parser.parse_args()

    if args.pre_norm:
        dataset = load_dataset(args.dataset, transform = T.NormalizeFeatures())
    else:
        dataset = load_dataset(args.dataset)

    cluster = Clustering(dataset.y, dataset.num_classes, args.n_seeds, args.path)

    accs = []
    nmis = []
    f1s = []
    for s in range(args.n_model.seeds):
        if args.load_model:
            path = args.path+"_split_0_seed_{}.pt".format(s)
            model = Encoder(dataset.num_features, args.hidden, dataset.num_classes, args.normalization)
            z = cluster.load_model(model, dataset[0], path)
            acc, nmi, f1 = cluster(z)
        else:
            path = args.path+"_split_0_seed_{}.npy".format(s)
            z = cluster.load_embed(path)
            acc, nmi, f1 = cluster(z)

        accs.extend(acc)
        nmis.extend(nmi)
        f1s.extend(f1)

    print("acc: ", accs)
    print("nmi: ", nmis)
    print("f1: ", f1s)
    print("Acc mean: {:.4f}, std: {:.4f}".format(np.mean(accs), np.std(accs)))
    print("Nmi mean: {:.4f}, std: {:.4f}".format(np.mean(nmis), np.std(nmis)))
    print("F1 macro mean: {:.4f}, std: {:.4f}".format(np.mean(f1s), np.std(f1s)))



