from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier


def get_model(args):
    if args.model == 'lr':
        return LogisticRegression(warm_start=True, solver="liblinear")
    if args.model == 'svm':
        return SVC(C=args.c, kernel=args.kernel)
    if args.model == 'gb':
        return GradientBoostingClassifier(n_estimators=100, subsample=0.8)
