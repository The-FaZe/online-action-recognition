from sklearn.metrics import average_precision_score, confusion_matrix
import argparse
import sys
import numpy as np


def softmax(raw_score, T=1):
    exp_s = np.exp((raw_score - raw_score.max(axis=-1)[..., None])*T)
    sum_s = exp_s.sum(axis=-1)
    return exp_s / sum_s[..., None]

def default_aggregation_func(score_arr, normalization=True, crop_agg=None):
    """
    This is the default function for make video-level prediction
    :param score_arr: a 3-dim array with (frame, crop, class) layout
    :return:
    """
    crop_agg = np.mean if crop_agg is None else crop_agg
    if normalization:
        return softmax(crop_agg(score_arr, axis=1).mean(axis=0))
    else:
        return crop_agg(score_arr, axis=1).mean(axis=0)
    
def mean_class_accuracy(scores, labels):
    pred = np.argmax(scores, axis=1)
    cf = confusion_matrix(labels, pred).astype(float)

    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)

    return np.mean(cls_hit/cls_cnt)


def fusion():
    
  parser = argparse.ArgumentParser()
  parser.add_argument('score_files', nargs='+', type=str)
  parser.add_argument('--score_weights', nargs='+', type=float, default=None)
  parser.add_argument('--crop_agg', type=str, choices=['max', 'mean'], default='mean')
  args = parser.parse_args()

  score_npz_files = [np.load(x) for x in args.score_files]

  if args.score_weights is None:
      score_weights = [1] * len(score_npz_files)
  else:
      score_weights = args.score_weights
      if len(score_weights) != len(score_npz_files):
          raise ValueError("Only {} weight specifed for a total of {} score files"
                           .format(len(score_weights), len(score_npz_files)))

  score_list = [x['scores'][:, 0] for x in score_npz_files]
  label_list = [x['labels'] for x in score_npz_files]

  # label verification

  # score_aggregation
  agg_score_list = []
  for score_vec in score_list:
      agg_score_vec = [default_aggregation_func(x, normalization=False, crop_agg=getattr(np, args.crop_agg)) for x in score_vec]
      agg_score_list.append(np.array(agg_score_vec))

  final_scores = np.zeros_like(agg_score_list[0])
  for i, agg_score in enumerate(agg_score_list):
      final_scores += agg_score * score_weights[i]

  # accuracy
  acc = mean_class_accuracy(final_scores, label_list[0])
  print ('Final accuracy {:02f}%'.format(acc * 100))
  
if __name__ == "__main__":
    fusion()