use crate::{
    optim::SimpleOptimizer,
    record::{Record, RecordSettings},
};
use burn_tensor::backend::Backend;
use core::any::Any;
use serde::{Deserialize, Serialize};

pub enum AdaptorRecordV1<O: SimpleOptimizer<B>, B: Backend> {
    Rank1(O::State<1>),
    Rank2(O::State<2>),
    Rank3(O::State<3>),
    Rank4(O::State<4>),
    Rank5(O::State<5>),
    Rank6(O::State<6>),
    Rank7(O::State<7>),
    Rank8(O::State<8>),
}

impl<O: SimpleOptimizer<B>, B: Backend> Clone for AdaptorRecordV1<O, B> {
    fn clone(&self) -> Self {
        match self {
            AdaptorRecordV1::Rank1(record) => AdaptorRecordV1::Rank1(record.clone()),
            AdaptorRecordV1::Rank2(record) => AdaptorRecordV1::Rank2(record.clone()),
            AdaptorRecordV1::Rank3(record) => AdaptorRecordV1::Rank3(record.clone()),
            AdaptorRecordV1::Rank4(record) => AdaptorRecordV1::Rank4(record.clone()),
            AdaptorRecordV1::Rank5(record) => AdaptorRecordV1::Rank5(record.clone()),
            AdaptorRecordV1::Rank6(record) => AdaptorRecordV1::Rank6(record.clone()),
            AdaptorRecordV1::Rank7(record) => AdaptorRecordV1::Rank7(record.clone()),
            AdaptorRecordV1::Rank8(record) => AdaptorRecordV1::Rank8(record.clone()),
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub enum AdaptorRecordItemV1<O: SimpleOptimizer<B>, B: Backend, S: RecordSettings> {
    Rank1(<O::State<1> as Record>::Item<S>),
    Rank2(<O::State<2> as Record>::Item<S>),
    Rank3(<O::State<3> as Record>::Item<S>),
    Rank4(<O::State<4> as Record>::Item<S>),
    Rank5(<O::State<5> as Record>::Item<S>),
    Rank6(<O::State<6> as Record>::Item<S>),
    Rank7(<O::State<7> as Record>::Item<S>),
    Rank8(<O::State<8> as Record>::Item<S>),
}

impl<O, B> AdaptorRecordV1<O, B>
where
    O: SimpleOptimizer<B>,
    B: Backend,
{
    pub fn into_state<const D: usize>(self) -> O::State<D> {
        let boxed_state: Box<dyn Any> = match self {
            AdaptorRecordV1::Rank1(s) => Box::new(s),
            AdaptorRecordV1::Rank2(s) => Box::new(s),
            AdaptorRecordV1::Rank3(s) => Box::new(s),
            AdaptorRecordV1::Rank4(s) => Box::new(s),
            AdaptorRecordV1::Rank5(s) => Box::new(s),
            AdaptorRecordV1::Rank6(s) => Box::new(s),
            AdaptorRecordV1::Rank7(s) => Box::new(s),
            AdaptorRecordV1::Rank8(s) => Box::new(s),
        };
        let state = boxed_state
            .downcast::<O::State<D>>()
            .expect("Unsupported state dimension, dimension up to 8 are supported.");
        *state
    }
    pub fn from_state<const D: usize>(state: O::State<D>) -> Self {
        let state: Box<dyn Any> = Box::new(state);

        match D {
            1 => AdaptorRecordV1::Rank1(*state.downcast().unwrap()),
            2 => AdaptorRecordV1::Rank2(*state.downcast().unwrap()),
            3 => AdaptorRecordV1::Rank3(*state.downcast().unwrap()),
            4 => AdaptorRecordV1::Rank4(*state.downcast().unwrap()),
            5 => AdaptorRecordV1::Rank5(*state.downcast().unwrap()),
            6 => AdaptorRecordV1::Rank6(*state.downcast().unwrap()),
            7 => AdaptorRecordV1::Rank7(*state.downcast().unwrap()),
            8 => AdaptorRecordV1::Rank8(*state.downcast().unwrap()),
            _ => panic!("Unsupported state dimension, dimension up to 8 are supported."),
        }
    }
}

impl<O, B> Record for AdaptorRecordV1<O, B>
where
    O: SimpleOptimizer<B>,
    B: Backend,
{
    type Item<S: RecordSettings> = AdaptorRecordItemV1<O, B, S>;

    fn into_item<S: RecordSettings>(self) -> Self::Item<S> {
        match self {
            AdaptorRecordV1::Rank1(record) => AdaptorRecordItemV1::Rank1(record.into_item()),
            AdaptorRecordV1::Rank2(record) => AdaptorRecordItemV1::Rank2(record.into_item()),
            AdaptorRecordV1::Rank3(record) => AdaptorRecordItemV1::Rank3(record.into_item()),
            AdaptorRecordV1::Rank4(record) => AdaptorRecordItemV1::Rank4(record.into_item()),
            AdaptorRecordV1::Rank5(record) => AdaptorRecordItemV1::Rank5(record.into_item()),
            AdaptorRecordV1::Rank6(record) => AdaptorRecordItemV1::Rank6(record.into_item()),
            AdaptorRecordV1::Rank7(record) => AdaptorRecordItemV1::Rank7(record.into_item()),
            AdaptorRecordV1::Rank8(record) => AdaptorRecordItemV1::Rank8(record.into_item()),
        }
    }

    fn from_item<S: RecordSettings>(item: Self::Item<S>) -> Self {
        match item {
            AdaptorRecordItemV1::Rank1(item) => {
                AdaptorRecordV1::Rank1(<O::State<1> as Record>::from_item(item))
            }
            AdaptorRecordItemV1::Rank2(item) => {
                AdaptorRecordV1::Rank2(<O::State<2> as Record>::from_item(item))
            }
            AdaptorRecordItemV1::Rank3(item) => {
                AdaptorRecordV1::Rank3(<O::State<3> as Record>::from_item(item))
            }
            AdaptorRecordItemV1::Rank4(item) => {
                AdaptorRecordV1::Rank4(<O::State<4> as Record>::from_item(item))
            }
            AdaptorRecordItemV1::Rank5(item) => {
                AdaptorRecordV1::Rank5(<O::State<5> as Record>::from_item(item))
            }
            AdaptorRecordItemV1::Rank6(item) => {
                AdaptorRecordV1::Rank6(<O::State<6> as Record>::from_item(item))
            }
            AdaptorRecordItemV1::Rank7(item) => {
                AdaptorRecordV1::Rank7(<O::State<7> as Record>::from_item(item))
            }
            AdaptorRecordItemV1::Rank8(item) => {
                AdaptorRecordV1::Rank8(<O::State<8> as Record>::from_item(item))
            }
        }
    }
}