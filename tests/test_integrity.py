"""Integrity tests for leakage checks, threshold rules, and model interfaces."""
import unittest
import numpy as np
import pandas as pd
import torch
from ecg_project.data import partition_masks
from ecg_project.experiment import thresholds_from_validation
from ecg_project.models import build_model,parameter_count


class IntegrityTests(unittest.TestCase):
    def test_patient_overlap_is_rejected(self):
        frame=pd.DataFrame({'strat_fold':[1,9,10],'patient_id':[10,20,10]})
        with self.assertRaisesRegex(ValueError,'Patient leakage'):partition_masks(frame)

    def test_patient_disjoint_partitions_are_preserved(self):
        frame=pd.DataFrame({'strat_fold':[1,8,9,10],'patient_id':[10,10,20,30]})
        masks=partition_masks(frame)
        self.assertEqual([int(masks[k].sum()) for k in ['train','validation','test']],[2,1,1])

    def test_threshold_ties_select_largest_allowed_value(self):
        y=np.zeros((6,5));p=np.full((6,5),0.4)
        np.testing.assert_allclose(thresholds_from_validation(y,p),np.full(5,0.95))

    def test_models_produce_five_finite_logits_and_update(self):
        torch.set_num_threads(2)
        for name in ['compact_cnn','resnet1d','inception_time']:
            torch.manual_seed(42);model=build_model(name)
            x=torch.randn(4,12,1000);target=torch.randint(0,2,(4,5)).float()
            optimizer=torch.optim.AdamW(model.parameters(),lr=0.001)
            before=next(model.parameters()).detach().clone()
            logits=model(x)
            self.assertEqual(logits.shape,(4,5));self.assertTrue(torch.isfinite(logits).all())
            loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,target)
            loss.backward();optimizer.step()
            self.assertFalse(torch.equal(before,next(model.parameters()).detach()))
        self.assertEqual(parameter_count(build_model('compact_cnn')),221573)


if __name__=='__main__':unittest.main()
