#[allow(unused_macros)]
macro_rules! debug {
  ($($e:expr), *) => {println!(concat!($(stringify!($e), " = {:?}\n"), *), $($e), *)}
}
 
fn main() {
  const TIME_LIMIT: f64 = 9.99;
  let t = Time::new();
  
  let (v, e) = get::tuple::<usize,usize>();
  let es = get::tuple3s::<usize,usize,isize>(e);
  let (v_emb, e_emb) = get::tuple::<usize,usize>();
  let es_emb = get::tuples::<usize,usize>(e_emb);
  
  let n_emb = (v_emb as f64).sqrt() as usize;
  let n = (v as f64).sqrt().ceil() as usize;
  let central = if n_emb%2==0 {n_emb*n_emb/2 - n_emb/2} else {(n_emb*n_emb+1) / 2};
  
  let g = Graph::new_labeled(v, &es);
  let g_emb = Graph::new_nonlabeled(v_emb, &es_emb);
  
  let dom = sort_by_radius(&g).into_iter().chain((v+1..v+4*n+1)).collect::<Vec<_>>();
  let img = g_emb.bfs(central).into_iter().collect::<Vec<_>>();
  let f = dom.iter().cloned().zip(img.iter().cloned()).collect::<Vec<_>>();
  let mut bj = Bijection::new(&f);
  let mut bj_f = bj.clone();
 
  let mut score_max = score(&es, &g_emb, &bj);
  let mut score_f = score_max;
  
  let mut count = 0;
  
  let mut rand = XorShift::new_seed(&t);
  // let nf = f.len();
  let ve = std::cmp::min(v_emb, v+4*n);
  while t.elapsed() < TIME_LIMIT {
    // let i = rand.from_to(0, (nf-2) as u64) as usize;
    // let u1 = bj.source(img[i]);
    // let u2 = bj.source(img[i+1]);
    let u1 = rand.from_to(1, ve as u64) as usize;
    let u2 = rand.from_to(1, ve as u64) as usize;
    bj.swap_dom(u1, u2);
    let score = score(&es, &g_emb, &bj);
    if score_f < score {
      score_f = score;
      bj_f = bj.clone()
    }
    let pt = (prop_transit(score_max, score, count) * 100.0) as u64;
    // if count % 10000 == 0 {debug!(pt)}
    if rand.from_to(1, 99) < pt {
      score_max = score
    } else {
      bj.swap_dom(u1, u2)
    }
    count += 1
  }
  
  for i in 1 .. v+1 {
    println!("{} {}", i, bj_f.target(i))
  }
  
  // debug!(score_f, count)
}
 
fn score(es: &Vec<(usize,usize,isize)>, g_emb: &Graph, bj: &Bijection) -> isize {
  es.iter().map(|&(u,v,w)| if g_emb.is_adjacent(bj.map[u].unwrap(),bj.map[v].unwrap()) {w} else {0}).sum()
}
 
fn prop_transit(e0: isize, e1: isize, t: usize) -> f64 {
  // let exp = |x| 2.7182818284f64.powf(x);
  // if e0 < e1 {1.0} else {exp(((e1-e0)*100000) as f64 / std::cmp::max(0, 500000 - t as isize) as f64)}
  if e0 <= e1 {1.0} else {1.0 / ((e0-e1) * (1 + t / 10000) as isize) as f64}
}
 
fn sort_by_radius(g: &Graph) -> Vec<usize> {
  let mut buf = vec![];
  for v in 1 .. g.size + 1 {
    buf.push((g.ultimate(v), -g.adj[v].iter().map(|&(_,w)|w).sum::<isize>(), v))
  }
  buf.sort();
  buf.into_iter().map(|(_,_,v)|v).collect()
}
 
/*
fn sort_by_degweight(g: &Graph) -> Vec<usize> {
  let mut buf = g.adj.iter().map(|v|v.iter().map(|&(_,w)|w).sum::<isize>()).zip(0..).collect::<Vec<_>>();
  //let mut buf = g.adj.iter().map(|v|v.len()).zip(0..).collect::<Vec<_>>();
  buf.sort_by(|x,y|y.cmp(&x));
  buf.iter().map(|&(_,i)|i).collect()
}
*/
 
/*
fn take_rand(vec: &Vec<usize>, n: usize) -> Vec<usize> {
  let start = Instant::now();
  let rand = |a,b| start.elapsed().subsec_nanos() as usize % (b-a+1) + a;
  let mut buf = vec![];
  let mut took = vec![false; vec.len()];
  for _ in 0 .. n {
    let mut i = rand(0usize, vec.len() - 1);
    while took[i] {i = (i+1)%vec.len()}
    buf.push(vec[i]);
    took[i] = true
  }
  buf
}
*/
 
use std::time::*;
struct Time {
  start: Instant
}
 
#[allow(dead_code)]
impl Time {
  fn new() -> Time {
    Time {start: Instant::now()}
  }
 
  fn elapsed(&self) -> f64 {
    let elapsed = self.start.elapsed();
    let sec = elapsed.as_secs() as f64;
    let nano = elapsed.subsec_nanos() as f64 / 1000000000.0;
    sec + nano
  }
}
 
#[derive(Copy, Clone)]
struct XorShift {
  state1: u64,
  state2: u64
}
 
#[allow(dead_code)]
impl XorShift {
  fn new() -> XorShift {
    use std::time::*;
    let start = Instant::now();
    let seed1 = start.elapsed().subsec_nanos() as u64;
    let seed2 = start.elapsed().subsec_nanos() as u64;
    XorShift {state1: seed1, state2: seed2}
  }
 
  fn new_seed(t: &Time) -> XorShift {
    let seed1 = t.start.elapsed().subsec_nanos() as u64;
    let seed2 = t.start.elapsed().subsec_nanos() as u64;
    XorShift {state1: seed1, state2: seed2}
  }
 
  fn next(&mut self) -> u64 {
    let mut s1 = self.state2;
    let mut s2 = self.state1;
    s1 = s1 ^ (s1 >> 26);
    s2 = s2 ^ (s2 << 23);
    s2 = s2 ^ (s2 >> 17);
    self.state1 = self.state2;
    self.state2 = s1 ^ s2;
    (self.state1 >> 1) + (self.state2 >> 1)
  }
  
  fn from_to(&mut self, from: u64, to: u64) -> u64 {
    self.next() % (to - from + 1) + from
  }
}
 
#[derive(Clone, Debug)]
struct Bijection {
  n: usize,
  map: Vec<Option<usize>>,
  inv: Vec<Option<usize>>
}
 
#[allow(dead_code)]
impl Bijection {
  fn new(f: &[(usize, usize)]) -> Bijection {
    use std::collections::HashSet;
    
    let dom = f.iter().map(|f|f.0).collect::<HashSet<_>>();
    let img = f.iter().map(|f|f.1).collect::<HashSet<_>>();
    assert!(dom.len() == f.len());
    assert!(img.len() == f.len());
 
    let max_dom = dom.iter().max().unwrap();
    let max_img = img.iter().max().unwrap();
    let mut map = vec![None; max_dom+1];
    let mut inv = vec![None; max_img+1];
    for &(x, y) in f {
      map[x] = Some(y);
      inv[y] = Some(x)
    }
    
    Bijection{n: f.len(), map: map, inv: inv}
  }
  
  fn target(&self, x: usize) -> usize {
    self.map[x].unwrap()
  }
 
  fn source(&self, y: usize) -> usize {
    self.inv[y].unwrap()
  }
 
  fn swap_dom(&mut self, x1: usize, x2: usize) {
    assert!(self.map[x1].is_some());
    assert!(self.map[x2].is_some());
    let y1 = self.map[x1].unwrap();
    let y2 = self.map[x2].unwrap();
    self.map.swap(x1, x2);
    self.inv.swap(y1, y2)
  }
  
  fn swap_img(&mut self, y1: usize, y2: usize) {
    assert!(self.inv[y1].is_some());
    assert!(self.inv[y2].is_some());
    let x1 = self.inv[y1].unwrap();
    let x2 = self.inv[y2].unwrap();
    self.inv.swap(y1, y2);
    self.map.swap(x1, x2)
  }
  
  fn print(&self) {
    for (x, &opty) in self.map.iter().enumerate() {
      if let Some(y) = opty {println!("{} {}", x, y)}
    }
  }
}
 
struct Graph {
  size: usize,
  adj: Vec<Vec<(usize, isize)>>
}
 
impl Clone for Graph {
  fn clone(&self) -> Graph {
    Graph {
      size: self.size,
      adj: self.adj.clone()
    }
  }
}
 
#[allow(dead_code)]
impl Graph {
  fn new_nonlabeled(n: usize, edges: &[(usize, usize)]) -> Graph {
    let edges = edges.iter().map(|&(a,b)|(a,b,1)).collect::<Vec<_>>();
    Graph::new_labeled(n, &edges)
  }
 
  fn new_labeled(n: usize, edges: &[(usize, usize, isize)]) -> Graph {
    let mut g: Vec<Vec<(usize, isize)>> = vec![vec![]; n+1];
    for &(a, b, c) in edges {
      g[a].push((b,c));
      g[b].push((a,c));  // delete for digraph
    }
    Graph {size: n, adj: g}
  }
 
  fn edges(&self) -> Vec<(usize, usize, isize)> {
    let mut buf = vec![];
    for (i, next) in self.adj.iter().enumerate() {
      for &(j, x) in next {
        buf.push((i, j, x))
      }
    }
    buf
  }
 
  fn is_adjacent(&self, u: usize, v: usize) -> bool {
    self.adj[u].iter().any(|&(w,_)|w==v) || self.adj[v].iter().any(|&(w,_)|w==u)
  }
 
  fn is_connected(&self) -> bool {
    self.dfs(1).len() == self.size
  }
 
  fn is_hitofude(&self) -> bool {
    let deg_odd = self.adj.iter().filter(|vs|vs.len()%2==1).count();
    deg_odd == 0 || deg_odd == 2
  }
 
  fn bellman_ford(&self, s: usize) -> Vec<isize> {
    const INF: isize = std::isize::MAX >> 1;
    let edges = self.edges();
    let mut bf: Vec<isize> = vec![INF; self.size];
    bf[s] = 0;
 
    for _ in 1 .. self.size {
      for &(v, w, c) in &edges {
        bf[w] = std::cmp::min(bf[w], bf[v]+c)
      }
    }
    bf
  }
 
  fn dijkstra(&self, s: usize) -> Vec<isize> {
    use std::collections::BinaryHeap;
 
    const INF: isize = std::isize::MAX;
    let mut dk: Vec<isize> = vec![INF; self.size];
    dk[s] = 0;
 
    let mut pq = BinaryHeap::new();
    pq.push((0,s));
 
    while let Some((acc,v)) = pq.pop() {
      let acc = -acc;
      for &(w,c) in &self.adj[v] {
        let cand = acc + c;
        if cand < dk[w] {
          dk[w] = cand;
          pq.push((-cand,w));
        }
      }
    }
    dk
  }
 
  fn warshall_floyd(&self) -> Vec<Vec<isize>> {
    const INF: isize = std::isize::MAX >> 1;
    let mut wf: Vec<Vec<isize>> = vec![vec![INF; self.size+1]; self.size+1];
    for i in 1 .. self.size+1 {wf[i][i] = 0}
 
    for (next, i) in self.adj.iter().zip(0..) {
      for &(j, x) in next {
        wf[i][j] = x
      }
    }
    for k in 1 .. self.size+1 {
      for i in 1 .. self.size+1 {
        for j in 1 .. self.size+1 {
          wf[i][j] = std::cmp::min(wf[i][j], wf[i][k] + wf[k][j]);
        }
      }
    }
    wf
  }
 
  fn coloring2(&self) -> Option<(Vec<usize>, Vec<usize>)> {
    fn paint(v: usize, p: bool, g: &Graph, cv: &mut Vec<Option<bool>>) -> bool {
      match cv[v] {
        None => {
          let next = &g.adj[v];
          cv[v] = Some(p);
          next.iter().all(|&(w,_)|paint(w,!p,g,cv))},
        Some(q) => {q == p}
      }
    }
 
    let mut canvas: Vec<Option<bool>> = vec![None; self.size+1];
    let ans = paint(1, false, self, &mut canvas);
    let bs = canvas.iter().enumerate().filter(|&(_,&v)|v==Some(false)).map(|(i,_)|i).collect::<Vec<_>>();
    let ws = canvas.iter().enumerate().filter(|&(_,&v)|v==Some(true)).map(|(i,_)|i).collect::<Vec<_>>();
    if ans {Some((bs,ws))} else {None}
  }
 
  fn dfs(&self, s: usize) -> Vec<usize> {
    fn go(g: &Graph, current: usize, mut path: &mut Vec<usize>, mut visited: &mut Vec<bool>) {
      for &(next, _) in &g.adj[current] {
        if visited[next] {
          continue
        } else {
          visited[next] = true;
          path.push(next);
          go(&g, next, &mut path, &mut visited)
        }
      }
    }
 
    let mut path = vec![s];
    let mut visited = vec![false; self.size+1];
    visited[s] = true;
    go(&self, s, &mut path, &mut visited);
    path
  }
 
  fn bfs(&self, v: usize) -> Vec<usize> {
    use std::collections::VecDeque;
    let mut q = VecDeque::new();
    let mut path = vec![];
    let mut vd = vec![false; self.size+1];
    vd[v] = true;
    q.push_back(v);
    while let Some(w) = q.pop_front() {
      path.push(w);
      for &(next,_) in self.adj[w].iter() {
        if !vd[next] {
          q.push_back(next);
          vd[next] = true
        }
      }
    }
    path
  }
 
  fn ultimate(&self, v: usize) -> usize {
    use std::collections::VecDeque;
    let mut q = VecDeque::new();
    let mut dist = 0;
    let mut vd = vec![false; self.size+1];
    vd[v] = true;
    q.push_back((v,dist));
    while let Some((w, d)) = q.pop_front() {
      dist = std::cmp::max(dist, d);
      for &(next,_) in self.adj[w].iter() {
        if !vd[next] {
          q.push_back((next, d+1));
          vd[next] = true
        }
      }
    }
    dist
  }
 
  fn cut(&mut self, v: usize, w: usize) {
    self.adj[v].retain(|&(t,_)| t != w);
    self.adj[w].retain(|&(t,_)| t != v);
  }
}
 
#[allow(dead_code)]
mod get {
  use std::io::*;
  use std::str::*;
 
  pub fn val<T: FromStr>() -> T {
    let mut buf = String::new();
    let s = stdin();
    s.lock().read_line(&mut buf).ok();
    buf.trim_right().parse::<T>().ok().unwrap()
  }
 
  pub fn vals<T: FromStr>(n: usize) -> Vec<T> {
    let mut vec: Vec<T> = vec![];
    for _ in 0 .. n {
      vec.push(val());
    }
    vec
  }
 
  pub fn tuple<T1: FromStr, T2: FromStr>() -> (T1, T2) {
    let mut buf = String::new();
    let s = stdin();
    s.lock().read_line(&mut buf).ok();
    let mut it = buf.trim_right().split_whitespace();
    let x = it.next().unwrap().parse::<T1>().ok().unwrap();
    let y = it.next().unwrap().parse::<T2>().ok().unwrap();
    (x, y)
  }
 
  pub fn tuples<T1: FromStr, T2: FromStr>(n: usize) -> Vec<(T1, T2)> {
    let mut vec: Vec<(T1, T2)> = vec![];
    for _ in 0 .. n {
      vec.push(tuple());
    }
    vec
  }
 
  pub fn tuple3<T1: FromStr, T2: FromStr, T3: FromStr>() -> (T1, T2, T3) {
    let mut buf = String::new();
    let s = stdin();
    s.lock().read_line(&mut buf).ok();
    let mut it = buf.trim_right().split_whitespace();
    let x = it.next().unwrap().parse::<T1>().ok().unwrap();
    let y = it.next().unwrap().parse::<T2>().ok().unwrap();
    let z = it.next().unwrap().parse::<T3>().ok().unwrap();
    (x, y, z)
  }
 
  pub fn tuple3s<T1: FromStr, T2: FromStr, T3: FromStr>(n: usize) -> Vec<(T1, T2, T3)> {
    let mut vec: Vec<(T1, T2, T3)> = vec![];
    for _ in 0 .. n {
      vec.push(tuple3());
    }
    vec
  }
 
  pub fn list<T: FromStr>() -> Vec<T> {
    let mut buf = String::new();
    let s = stdin();
    s.lock().read_line(&mut buf).ok();
    buf.trim_right().split_whitespace().map(|t| t.parse::<T>().ok().unwrap()).collect()
  }
 
  pub fn lists<T: FromStr>(h: usize) -> Vec<Vec<T>> {
    let mut mat: Vec<Vec<T>> = vec![];
    for _ in 0 .. h {
      mat.push(list());
    }
    mat
  }
 
  pub fn chars() -> Vec<char> {
    let mut buf = String::new();
    let s = stdin();
    s.lock().read_line(&mut buf).ok();
    buf.trim_right().chars().collect()
  }
}
