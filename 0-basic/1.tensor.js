// 从一个多维数组创建一个rank-2的张量矩阵
const a = tf.tensor([[1, 2], [3, 4]]);
console.log('shape:', a.shape);
a.print();
// 或者您可以用一个一维数组并指定特定的形状来创建一个张量
const shape = [2, 2];
const b = tf.tensor([1, 2, 3, 4], shape);
console.log('shape:', b.shape);
b.print();

const c = tf.tensor([[1, 2], [3, 4]]);
console.log('c shape:', a.shape);
c.print();

const d = c.reshape([4, 1]);
console.log('d shape:', d.shape);
d.print();

// 整體數值操作
const x = tf.tensor([1, 2, 3, 4]);
const y = x.square();  // 相当于 tf.square(x)
y.print();

// 記憶體釋放
const e = tf.tensor([[1, 2], [3, 4]]);
const f = tf.tidy(() => {
  const result = e.square().log().neg();
  return result;
});