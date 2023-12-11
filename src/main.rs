use tch::nn::{Module, OptimizerConfig, embedding, EmbeddingConfig};
use tch::{kind, nn, Device, Tensor, Kind};

const WORDS: i64 = 10;
fn attention(p: nn::Path, headd: i64, embd: i64, seqd: i64) -> impl nn::Module{
    let embeddings = embedding(&p, WORDS, embd, EmbeddingConfig::default());

    let kw = p.randn_standard("key projection",   &[embd, headd]);
    let qw = p.randn_standard("query projection", &[embd, headd]);
    let vw = p.randn_standard("value projection", &[embd, headd]);
    let proj = nn::linear(p, headd, 1, nn::LinearConfig::default());
    
    nn::func(move |x|{
        let k = x.apply(&embeddings).matmul(&kw);
        let q = x.apply(&embeddings).matmul(&qw);
        let v = x.apply(&embeddings).matmul(&vw);
        // seqd x headd
        
        let scores = k.matmul(&q.transpose(0,1)).softmax(-1, Kind::Float);
        scores.matmul(&v).apply(&proj)
    })
}

fn main(){
    let vs = nn::VarStore::new(Device::Cpu);
    let attn = attention(vs.root(), 6, 8, WORDS);
    let mut opt = nn::Adam::default().build(&vs, 1e-3).unwrap();
    for i in 0..10000{
        let xs: Vec<i64> = (0..WORDS-1).map(|n| if n == i%WORDS {0}else{n+1} ).collect();
        let ys: Vec<f32> = (0..WORDS-1).map(|n| if n == i%WORDS {1.}else{0.} ).collect();
        let xs = Tensor::from_slice(&xs);
        let ys = Tensor::from_slice(&ys);
        let loss = (attn.forward(&xs) - ys).pow_tensor_scalar(2).sum(Kind::Float);
        opt.backward_step(&loss);
        //println!("{loss:?}");
    }

    println!("{}", attn.forward(&Tensor::from_slice(&[1, 2, 0, 4, 5, 6, 7, 8, 9])));
}
