import torch
import numpy as np


class Evaluation:
    @staticmethod
    def r2_scores(predictions: torch.Tensor, targets: torch.Tensor, step_width: float, lle: float, limit: float = 0.9):
        output_tokens = predictions.shape[0]
        lt_r2 = step_width * output_tokens * lle
        r2_scores = np.zeros(shape=predictions.shape[0])
        for step in range(output_tokens):
            predictions_until_step = predictions[:step + 1]
            target_until_step = targets[:step + 1]
            step_mean_r2_score = Evaluation.r2_score(predictions_until_step, target_until_step)
            if step_mean_r2_score < limit:
                lt_r2 = min(lt_r2, step_width * step * lle)
            r2_scores[step] = step_mean_r2_score
        return r2_scores, lt_r2

    @staticmethod
    def r2_score(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        if predictions.shape[0] == 1 and targets.shape[0] == 1:
            return 1.

        data_means = targets.mean(0).mean(0)

        a = ((predictions - targets)**2).sum(dim=0)
        b = ((targets - data_means)**2).sum(dim=0)
        return (1. - a / b).mean()

    @staticmethod
    def median_r2_scores(predictions: torch.Tensor, targets: torch.Tensor, step_width: float, lle: float, limit: float = 0.9):
        output_tokens = predictions.shape[0]
        lt_r2 = step_width * output_tokens * lle
        r2_scores = np.zeros(shape=predictions.shape[0])
        for step in range(output_tokens):
            predictions_until_step = predictions[:step + 1]
            target_until_step = targets[:step + 1]
            step_median_r2_score = Evaluation.median_r2_score(predictions_until_step, target_until_step)
            if step_median_r2_score < limit:
                lt_r2 = min(lt_r2, step_width * step * lle)
            r2_scores[step] = step_median_r2_score
        return r2_scores, lt_r2

    @staticmethod
    def median_r2_score(predictions: torch.Tensor, targets: torch.Tensor):
        data_means = targets.mean(0).mean(0)

        a = ((predictions - targets)**2).sum(dim=0)
        b = ((targets - data_means)**2).sum(dim=0)
        return (1. - a / b).mean(1).median()

    @staticmethod
    def mape(predictions, targets):
        return (abs((predictions - targets) / targets)).mean()

    @staticmethod
    def median_mape(predictions, targets):
        m1 = (abs((predictions - targets) / targets)).mean(2)
        m2 = np.median(m1, 1)
        m3 = m2.mean(0)
        return m3

    @staticmethod
    def relative_mse(predictions, targets):
        data_means = targets.mean(0).mean(0)

        a = ((predictions - targets)**2).sum(0)
        b = ((targets - data_means)**2).sum(0)
        return (a / b).mean().item()

    @staticmethod
    def rmses(predictions, targets):
        m1 = ((predictions - targets)**2).mean(2)**(1/2)
        return m1.mean(1).cpu().numpy()

    @staticmethod
    def rmse(predictions, targets):
        m1 = ((predictions - targets)**2).mean(2)**(1/2)
        return m1.mean().item()

    @staticmethod
    def median_rmse(predictions, targets):
        m1 = ((predictions - targets)**2).mean(2)**(1/2)
        m2 = m1.median(1).values
        m3 = m2.mean(0)
        return m3.item()

    @staticmethod
    def nrmse(predictions, targets, norm_value):
        norm_value = abs(norm_value)
        m1 = ((predictions - targets)**2).mean(2)**(1/2) / norm_value
        return m1.mean().item()

    @staticmethod
    def m_nrmse_of_steps(predictions, targets, norm_value, from_step=None, to_step=None):
        norm_value = abs(norm_value)
        m1 = ((predictions - targets)**2).mean(2)**(1/2) / norm_value
        m2 = m1[from_step: to_step].mean(0)
        m3 = m2.median(0).values
        return m3.item()

    @staticmethod
    def nrmse_of_steps(predictions, targets, norm_value, from_step=None, to_step=None):
        norm_value = abs(norm_value)
        m1 = ((predictions - targets)**2).mean(2)**(1/2) / norm_value
        m2 = m1.mean(1)[from_step: to_step].mean()
        return m2.item()

    @staticmethod
    def median_nrmse(predictions, targets, norm_value):
        norm_value = abs(norm_value)
        m1 = ((predictions - targets)**2).mean(2)**(1/2) / norm_value
        m2 = m1.mean(0)
        m3 = m2.median(0).values
        return m3.item()

    @staticmethod
    def median_nrmses(predictions, targets, norm_value):
        norm_value = abs(norm_value)
        m1 = ((predictions - targets)**2).mean(2)**(1/2) / norm_value
        m2 = m1.median(1).values
        return np.array(m2.cpu())
