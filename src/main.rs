use tch::nn::{Module, OptimizerConfig, embedding, EmbeddingConfig};
use tch::{kind, nn, Device, Tensor, Kind};

const WORDS: i64 = 10;
fn attention(p: nn::Path, headd: i64, embd: i64, seqd: i64) -> impl nn::Module{
    let embeddings = embedding(&p, WORDS, embd, EmbeddingConfig::default());

    let kw = p.randn_standard("key projection",   &[embd, headd]).set_requires_grad(true);
    let qw = p.randn_standard("query projection", &[embd, headd]).set_requires_grad(true);
    let vw = p.randn_standard("value projection", &[embd, headd]).set_requires_grad(true);
    let proja = nn::linear(&p, headd, headd, nn::LinearConfig::default());
    let projb = nn::linear(&p, headd, seqd, nn::LinearConfig::default());
    
    nn::func(move |x|{
        let k = x.apply(&embeddings).matmul(&kw);
        let q = x.apply(&embeddings).matmul(&qw);
        let v = x.apply(&embeddings).matmul(&vw);

        // seqd x headd
        let seqd = v.size()[0];
        let sum = Tensor::ones(&[seqd, 1], (Kind::Float, Device::Cpu));

        let scores = k.matmul(&q.transpose(0,1)).softmax(0, Kind::Float);

        // The relu is important, to make sure the model is not just summing negative one-hot
        // encodings for the input, as this gives the correct result when applying the softmax,
        // without the model learning the desired algorithm.
        scores.matmul(&v).apply(&proja).transpose(0,1).matmul(&sum).transpose(0,1).apply(&projb).relu().softmax(1, Kind::Float)
    })
    //nn::func(move |x|{
    //    let x = one_hot(x) * -1;

    //    let seqd = x.size()[0];
    //    let sum = Tensor::ones(&[seqd, 1], (Kind::Float, Device::Cpu));

    //    x.transpose(0,1).matmul(&sum).softmax(0, Kind::Float)
    //})
}

fn main(){
    let vs = nn::VarStore::new(Device::Cpu);
    let attn = attention(vs.root(), 6, 6, WORDS);
    let mut opt = nn::AdamW::default().build(&vs, 1e-3).unwrap();
    for i in 0..20000{
        let xs: Vec<i64> = (0..WORDS).filter_map(|j| if j == i%WORDS {None}else{Some(j)} ).collect();
        let ys: Vec<f32> = (0..WORDS).map(|j| if j == i%WORDS {1.}else{0.} ).collect();
        let xs = Tensor::from_slice(&xs);
        let ys = Tensor::from_slice(&ys);
        let loss = (attn.forward(&xs) - ys).pow_tensor_scalar(2).sum(Kind::Float);
        opt.backward_step(&loss);
        //println!("{loss:?}");
    }

    for i in 0..10{
        let x: Vec<i64> = (0..WORDS).filter_map(|j| if j == i {None}else{Some(j)} ).collect();
        let t1 = attn.forward(&Tensor::from_slice(&x));
        //println!("{}", t1);
        //println!("{:?}", t1.argmax(1, false));
        println!("{i}: {:?}", t1.argmax(1, false));
    }
}
