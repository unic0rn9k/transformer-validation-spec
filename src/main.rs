use tch::nn::{Module, OptimizerConfig, embedding, EmbeddingConfig};
use tch::{kind, nn, Device, Tensor, Kind};

const WORDS: i64 = 10;
fn attention(p: nn::Path, headd: i64, embd: i64, seqd: i64) -> impl nn::Module{
    let embeddings = embedding(&p, WORDS+1, embd, EmbeddingConfig::default());

    let kw = p.randn_standard("key projection",   &[embd, headd]);
    let qw = p.randn_standard("query projection", &[embd, headd]);
    let vw = p.randn_standard("value projection", &[embd, headd]);
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
        scores.matmul(&v).apply(&proja).transpose(0,1).matmul(&sum).transpose(0,1).apply(&projb).softmax(1, Kind::Float)
    })
}

fn main(){
    let vs = nn::VarStore::new(Device::Cpu);
    let attn = attention(vs.root(), 6, 8, WORDS);
    let mut opt = nn::Adam::default().build(&vs, 1e-3).unwrap();
    for i in 0..10000{
        let xs: Vec<i64> = (1..=WORDS).filter_map(|n| if n == i%WORDS {None}else{Some(n)} ).collect();
        let ys: Vec<f32> = (1..=WORDS).map(|n| if n == i%WORDS {1.}else{0.} ).collect();
        let xs = Tensor::from_slice(&xs);
        let ys = Tensor::from_slice(&ys);
        let loss = (attn.forward(&xs) - ys).pow_tensor_scalar(2).sum(Kind::Float);
        opt.backward_step(&loss);
        //println!("{loss:?}");
    }

    println!("{}", attn.forward(&Tensor::from_slice(&[1, 2, 4, 5, 6, 7, 8, 9, 10])));
}
