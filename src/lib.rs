use std::{cell::RefCell, iter::zip, ops::Mul, sync::Arc};

use rand::{thread_rng, Rng};

const PUNISH_BACK_PERCENT: f64 = 0.8;
const WEIGHT_INCREASE_RATE: f64 = 0.02;
const WEIGHT_DECREASE_RATE: f64 = 0.02;
const NEURON_SENSITIVITY: f64 = 0.5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeuronState {
    On,
    Off,
}

impl Mul<f64> for NeuronState {
    type Output = f64;

    fn mul(self, rhs: f64) -> Self::Output {
        match self {
            Self::On => rhs,
            Self::Off => 0.,
        }
    }
}

#[derive(Debug, Clone)]
struct Link {
    weight: Arc<RefCell<f64>>,
    punish_back: Arc<RefCell<bool>>,
    back: Arc<RefCell<Neuron>>,
    forward: Arc<RefCell<Neuron>>,
}

impl Link {
    fn new(weight: f64, back: Arc<RefCell<Neuron>>, forward: Arc<RefCell<Neuron>>) -> Self {
        Self {
            back,
            forward,
            weight: Arc::new(RefCell::new(weight)),
            punish_back: Arc::new(RefCell::new(false)),
        }
    }
}

#[derive(Debug, Clone)]
struct Neuron {
    back: Vec<Link>,
    forward: Vec<Link>,
    state: NeuronState,
    read: bool,
}

impl Default for Neuron {
    fn default() -> Self {
        Self {
            back: Vec::new(),
            forward: Vec::new(),
            state: NeuronState::Off,
            read: false,
        }
    }
}

impl Neuron {
    fn forward(&mut self) {
        let mut state_sum = 0f64;

        for link in &self.back {
            let mut back = link.back.borrow_mut();
            let weight = link.weight.borrow();
            back.read = true;
            state_sum += back.state * *weight;
        }

        if state_sum >= NEURON_SENSITIVITY {
            self.state = NeuronState::On;
        } else {
            self.state = NeuronState::Off;
        }
        self.read = false;
    }

    fn back(&mut self) {
        let mut correction_weight = 0f64;
        let mut weight_sum = 0f64;

        for link in &self.forward {
            let punish = *link.punish_back.borrow();
            let weight = *link.weight.borrow();

            weight_sum += weight;

            if punish {
                correction_weight += weight;
            }
        }

        let punish_back = ((correction_weight / weight_sum > PUNISH_BACK_PERCENT)
            && !self.back.is_empty())
            || self.forward.is_empty();

        match self.state {
            NeuronState::Off => {
                if punish_back {
                    for link in &self.back {
                        match link.back.borrow().state {
                            NeuronState::Off => {
                                if *link.weight.borrow() > 0. {
                                    *link.punish_back.borrow_mut() = true;
                                }
                            }
                            NeuronState::On => *link.weight.borrow_mut() += WEIGHT_INCREASE_RATE,
                        }
                    }
                }
            }
            NeuronState::On => {
                if punish_back {
                    for link in &self.back {
                        match link.back.borrow().state {
                            NeuronState::Off => {
                                if *link.weight.borrow() < 0. {
                                    *link.punish_back.borrow_mut() = true;
                                }
                            }
                            NeuronState::On => {
                                *link.weight.borrow_mut() -= WEIGHT_DECREASE_RATE;

                                if *link.weight.borrow() > 0. {
                                    *link.punish_back.borrow_mut() = true;
                                }
                            }
                        }
                    }
                } else {
                    for link in &self.forward {
                        if *link.punish_back.borrow() {
                            *link.weight.borrow_mut() -= WEIGHT_DECREASE_RATE;
                        }
                    }
                }
            }
        }

        for link in &self.forward {
            *link.punish_back.borrow_mut() = false;
        }
    }
}

#[derive(Debug)]
pub struct Network {
    layers: Vec<Vec<Arc<RefCell<Neuron>>>>,
}

impl Network {
    pub fn new(dimensions: &[usize]) -> Self {
        let mut rng = thread_rng();
        let mut layers = Vec::new();

        for &count in dimensions {
            let mut layer = Vec::with_capacity(count);

            for _ in 0..count {
                let neuron = Arc::new(RefCell::new(Neuron::default()));

                let flattened_layers = layers
                    .iter()
                    .flatten()
                    .map(|neuron: &Arc<RefCell<Neuron>>| neuron.to_owned());
                for prev_neuron in flattened_layers {
                    let link = Link::new(
                        rng.gen_range(-1.0..=1.0),
                        prev_neuron.clone(),
                        neuron.clone(),
                    );
                    neuron.borrow_mut().back.push(link.clone());
                    prev_neuron.borrow_mut().forward.push(link);
                }

                layer.push(neuron);
            }

            layers.push(layer)
        }

        Self { layers }
    }

    pub fn calc(&self, data: &[NeuronState]) -> Vec<NeuronState> {
        debug_assert_eq!(data.len(), self.layers[0].len());
        for (value, neuron) in zip(data, self.layers[0].iter()) {
            let mut neuron = neuron.borrow_mut();
            neuron.state = *value;
            neuron.read = false;
        }

        for layer in self.layers[1..].iter() {
            for neuron in layer {
                neuron.borrow_mut().forward();
            }
        }

        self.layers[self.layers.len() - 1]
            .iter()
            .map(|neuron| neuron.borrow().state)
            .collect()
    }

    pub fn fit(&self, data: &[NeuronState]) {
        for (expected, neuron) in zip(data, self.layers[self.layers.len() - 1].iter()) {
            let mut neuron = neuron.borrow_mut();
            if neuron.state != *expected {
                neuron.back();
            }
        }

        for layer in self.layers.iter().rev().skip(1) {
            for neuron in layer {
                let mut neuron = neuron.borrow_mut();
                neuron.back();
            }
        }
    }
}
