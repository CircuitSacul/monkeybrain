use monkeybrain::{Network, NeuronState};
// use neuroflow::{FeedForward, data::DataSet};

fn data(data: &[u8]) -> Vec<NeuronState> {
    data.iter()
        .map(|val| match val {
            1 => NeuronState::On,
            0 => NeuronState::Off,
            _ => unreachable!(),
        })
        .collect()
}

// fn fdata(data: &[u8]) -> Vec<f64> {
//     data.iter().map(|val| *val as f64).collect()
// }

pub fn main() {
    let traning_data: &[([u8; 4], [u8; 1])] = &[
        ([1, 0, 0, 0], [1]),
        ([0, 1, 0, 0], [1]),
        ([0, 0, 1, 0], [1]),
        ([0, 0, 0, 1], [1]),
        ([1, 1, 0, 0], [0]),
        ([0, 1, 1, 0], [0]),
        ([0, 0, 1, 1], [0]),
        ([1, 1, 1, 1], [0]),
    ];

    // setup
    let mnn = Network::new(&[4, 1, 1]);

    // training
    for _ in 0..1000 {
        for (inp, out) in traning_data {
            mnn.calc(&data(inp));
            mnn.fit(&data(out));
        }
    }

    let mut right = 0;
    for (inp, expected) in traning_data {
        let out = mnn.calc(&data(inp));

        if data(expected)[0] == out[0] {
            right += 1;
        }
    }
    dbg!(right / traning_data.len());
    dbg!(mnn.calc(&data(&[1, 0, 0, 1])));
    dbg!(mnn.calc(&data(&[0, 1, 0, 1])));
}
