use super::{
    confusion_matrix::ConfusionMatrix,
    state::{FormatOptions, NumericMetricState},
    ClassificationAverage, ClassificationInput, ClassificationMetric, Metric, MetricEntry,
    MetricMetadata, Numeric,
};
use burn_core::tensor::backend::Backend;
use core::marker::PhantomData;

/// The precision metric.
pub struct RecallMetric<B: Backend> {
    state: NumericMetricState,
    _b: PhantomData<B>,
    threshold: f64,
    average: ClassificationAverage,
}

impl<B: Backend> Default for RecallMetric<B> {
    /// Creates a new metric instance with default values.
    fn default() -> Self {
        Self {
            state: NumericMetricState::default(),
            _b: PhantomData,
            threshold: 0.5,
            average: ClassificationAverage::Micro,
        }
    }
}

impl<B: Backend> Metric for RecallMetric<B> {
    const NAME: &'static str = "Precision";
    type Input = ClassificationInput<B>;
    fn update(
        &mut self,
        input: &ClassificationInput<B>,
        _metadata: &MetricMetadata,
    ) -> MetricEntry {
        let [sample_size, _] = input.predictions.dims();

        let conf_mat = ConfusionMatrix::new(input, self.threshold, self.average);
        let agg_metric = conf_mat.clone().true_positive() / conf_mat.positive();
        let metric = self.average.to_averaged_metric(agg_metric);

        self.state.update(
            100.0 * metric,
            sample_size,
            FormatOptions::new(Self::NAME).unit("%").precision(2),
        )
    }

    fn clear(&mut self) {
        self.state.reset()
    }
}

impl<B: Backend> Numeric for RecallMetric<B> {
    fn value(&self) -> f64 {
        self.state.value()
    }
}

impl<B: Backend> ClassificationMetric<B> for RecallMetric<B> {
    fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    fn with_average(mut self, average: ClassificationAverage) -> Self {
        self.average = average;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::{ClassificationAverage, Metric, MetricMetadata, Numeric, RecallMetric};
    use crate::metric::ClassificationMetric;
    use crate::tests::{dummy_classification_input, ClassificationType, THRESHOLD};
    use crate::TestBackend;
    use approx::assert_relative_eq;
    use strum::IntoEnumIterator;

    #[test]
    fn test_precision() {
        for class_avg_type in ClassificationAverage::iter() {
            for classification_type in ClassificationType::iter() {
                let (input, target_diff) = dummy_classification_input(&classification_type);
                //tp/(tp+fn) = 1 - fn/(tp+fn)
                let mut metric = RecallMetric::<TestBackend>::default()
                    .with_threshold(THRESHOLD)
                    .with_average(class_avg_type);
                let _entry = metric.update(&input, &MetricMetadata::fake());

                //fn/(tp+fp+tn+fn) = fn/(tp+fn)(1 + negative/positive)
                let agg_false_negative_rate =
                    class_avg_type.aggregate_mean(target_diff.clone().equal_elem(1));
                let positive = input.targets.clone().int();
                let agg_negative =
                    class_avg_type.aggregate_sum(positive.clone().bool().bool_not());
                let agg_positive = class_avg_type.aggregate_sum(positive.bool());
                //1 - fn(1 + n/p)/(tp+fp+tn+fn) = 1 - fn/(tp+fn) = tp/(tp+fn)
                let test_precision = class_avg_type.to_averaged_metric(
                    -agg_false_negative_rate * (agg_negative / agg_positive + 1.0) + 1.0,
                );
                assert_relative_eq!(
                    metric.value(),
                    test_precision * 100.0,
                    max_relative = 1e-3
                );
            }
        }
    }
}