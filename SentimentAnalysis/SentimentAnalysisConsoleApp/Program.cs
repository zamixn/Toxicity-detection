using System;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace SentimentAnalysisConsoleApp
{
    internal static class Program
    {
        public class InputData
        {
            [LoadColumn(0)]
            public bool Label { get; set; }
            [LoadColumn(2)]
            public string Text { get; set; }
        }

        public class OutputData
        {
            [ColumnName("PredictedLabel")]
            public bool Prediction;
            public float Probability { get; set; }
            public float Score { get; set; }
        }

        static void Main(string[] args)
        {
            string dataPath = @"../../../../Data/dataSentiment_short.tsv";
            MLContext mlContext = new MLContext();
            IDataView dataView = mlContext.Data.LoadFromTextFile<InputData>(dataPath, hasHeader: true);

            var dataProcessPipeline = mlContext.Transforms.Text.FeaturizeText(
                outputColumnName: "Features", inputColumnName: nameof(InputData.Text));
            var trainer = mlContext.BinaryClassification.Trainers.FastTree(
                labelColumnName: "Label", featureColumnName: "Features");
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            Console.WriteLine("Training...");
            ITransformer trainedModel = trainingPipeline.Fit(dataView);
            Console.WriteLine("Training Done.");


            var predEngine = mlContext.Model.CreatePredictionEngine<InputData, OutputData>(trainedModel);
            string line;
            while ((line = Console.ReadLine()) != "")
            {
                InputData sampleStatement = new InputData { Text = line };
                //Score
                var resultprediction = predEngine.Predict(sampleStatement);
                Console.WriteLine("Text: {0} | Prediction: {1} | Probability: {2}", sampleStatement.Text, resultprediction.Prediction ? "Toxic" : "Safe", resultprediction.Probability);
            }

        }
    }
}