
/**
 * 從網路抓取資料
 */
async function getData() {
    // 從網路抓取資料
    const carsDataResponse = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json');
    const carsData = await carsDataResponse.json();
    const cleaned = carsData.map(car => ({
        mpg: car.Miles_per_Gallon,
        horsepower: car.Horsepower,
    }))
        .filter(car => (car.mpg != null && car.horsepower != null));
    return cleaned;
}

/**
 * 建立模型
 */
function createModel() {
    
    const model = tf.sequential();
    // 新增一個 輸入層, 該輸入層連接至 dense 隱藏層 
    //      params: inputShape: 1代表1個數字 , units 1 代表一個數字權重是1
    model.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }));
    // 建立輸出層
    //      params: units 1 代表要輸出一個數字
    model.add(tf.layers.dense({ units: 1, useBias: true }));
    return model;
}

/**
 * 將JSON資料轉換為「張量」, 並回傳相關最大最小值
 * @param {*} data 正常所見之 JSON資料
 */
function convertToTensor(data) {
    // 轉換成 2D「張量」 => (Why is 2D ?...)
    return tf.tidy(() => {
        // Step 1. 重新打散範本  
        tf.util.shuffle(data);

        // Step 2. 將資料轉換成「張量」
        const inputs = data.map(d => d.horsepower)
        const labels = data.map(d => d.mpg);
        //      轉換成 inputs.length(樣本數) 個 張量, 其中每個樣本有一個輸入
        const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
        const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

        //Step 3. 將資料正規化至範圍 0 - 1 的數值
        const inputMax = inputTensor.max();
        const inputMin = inputTensor.min();
        const labelMax = labelTensor.max();
        const labelMin = labelTensor.min();
        //      先將值減最小值, 並除上 最大-最小值 , 得到 0~1的值
        const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
        const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

        return {
            inputs: normalizedInputs,
            labels: normalizedLabels,
            // Return the min/max bounds so we can use them later.
            inputMax,
            inputMin,
            labelMax,
            labelMin,
        }
    });
}



async function trainModel(model, inputs, labels) {
    // 訓練
    model.compile({
        optimizer: tf.train.adam(),
        loss: tf.losses.meanSquaredError,
        metrics: ['mse'],
    });
    // 一次訓練中, 最大可以處理的樣本數
    const batchSize = 32;
    // 可以循環訓練幾次, 稱為「迭代」
    const epochs = 50;

    return await model.fit(inputs, labels, {
        batchSize,
        epochs,
        shuffle: true,
        callbacks: tfvis.show.fitCallbacks(
            { name: 'Training Performance' },
            ['loss', 'mse'],
            { height: 200, callbacks: ['onEpochEnd'] }
        )
    });
}


/**
 * 評估模型
 * @param {*} model 模型
 * @param {*} inputData JSON資料
 * @param {*} normalizationData 正規化張量資料
 */
function testModel(model, inputData, normalizationData) {
    // 利用0~1線性迴歸值當做輸入, 以獲得在該模型中每個區間獲得的預測值, 並畫出來

    const { inputMax, inputMin, labelMin, labelMax } = normalizationData;

    const [xs, preds] = tf.tidy(() => {

        const xs = tf.linspace(0, 1, 100); // 建立 0~1, 100 個線性迴歸值
        const preds = model.predict(xs.reshape([100, 1])); // 預測出 100 個線性迴歸的答案

        const unNormXs = xs 
            .mul(inputMax.sub(inputMin)) //將正規化數值反向運算, 獲得非正規化資料 => 乘(最大 - 最小) + 最小
            .add(inputMin);

        const unNormPreds = preds
            .mul(labelMax.sub(labelMin))
            .add(labelMin);

        // 返回非正規化（原型）模擬資料, 及預測之答案
        return [unNormXs.dataSync(), unNormPreds.dataSync()];
    });


    const predictedPoints = Array.from(xs).map((val, i) => {
        return { x: val, y: preds[i] }
    });

    const originalPoints = inputData.map(d => ({
        x: d.horsepower, y: d.mpg,
    }));


    tfvis.render.scatterplot(
        { name: 'Model Predictions vs Original Data' },
        { values: [originalPoints, predictedPoints], series: ['original', 'predicted'] },
        {
            xLabel: 'Horsepower',
            yLabel: 'MPG',
            height: 300
        }
    );
}


async function run() {
    // 主程式
    // 抓資料
    const data = await getData();
    const values = data.map(d => ({
        x: d.horsepower,
        y: d.mpg,
    }));
    // 顯示資料分布
    tfvis.render.scatterplot(
        { name: 'Horsepower v MPG' },
        { values },
        {
            xLabel: 'Horsepower',
            yLabel: 'MPG',
            height: 300
        }
    );
    // 建立模型
    const model = createModel();
    // 顯示開模型摘要
    tfvis.show.modelSummary({ name: 'Model Summary' }, model);
    // 轉換為「張量」
    const tensorData = convertToTensor(data);
    const { inputs, labels } = tensorData;
    // 訓練模型
    await trainModel(model, inputs, labels);
    console.log('Done Training');
    // 評估模型
    testModel(model, data, tensorData);
}



document.addEventListener('DOMContentLoaded', run);

