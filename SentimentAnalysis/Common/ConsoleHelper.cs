﻿using System;
using System.Linq;
using System.Collections.Generic;
using Microsoft.ML.Data;
using Microsoft.ML;
using static Microsoft.ML.TrainCatalogBase;

namespace Common
{
    public static class ConsoleHelper
    {
        public static void PrintPrediction(string prediction)
        {
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"Predicted : {prediction}");
            Console.WriteLine($"*************************************************");
        }

        public static void PrintRegressionPredictionVersusObserved(string predictionCount, string observedCount)
        {
            Console.WriteLine($"-------------------------------------------------");
            Console.WriteLine($"Predicted : {predictionCount}");
            Console.WriteLine($"Actual:     {observedCount}");
            Console.WriteLine($"-------------------------------------------------");
        }

        public static void PrintRegressionMetrics(string name, RegressionMetrics metrics)
        {
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Metrics for {name} regression model      ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       LossFn:        {metrics.LossFn:0.##}");
            Console.WriteLine($"*       R2 Score:      {metrics.RSquared:0.##}");
            Console.WriteLine($"*       Absolute loss: {metrics.L1:#.##}");
            Console.WriteLine($"*       Squared loss:  {metrics.L2:#.##}");
            Console.WriteLine($"*       RMS loss:      {metrics.Rms:#.##}");
            Console.WriteLine($"*************************************************");
        }

        public static void PrintBinaryClassificationMetrics(string name, CalibratedBinaryClassificationMetrics metrics)
        {
            Console.WriteLine($"************************************************************");
            Console.WriteLine($"*       Metrics for {name} binary classification model      ");
            Console.WriteLine($"*-----------------------------------------------------------");
            Console.WriteLine($"*       Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"*       Auc:      {metrics.Auc:P2}");
            Console.WriteLine($"*       Auprc:  {metrics.Auprc:P2}");
            Console.WriteLine($"*       F1Score:  {metrics.F1Score:P2}");
            Console.WriteLine($"*       LogLoss:  {metrics.LogLoss:#.##}");
            Console.WriteLine($"*       LogLossReduction:  {metrics.LogLossReduction:#.##}");
            Console.WriteLine($"*       PositivePrecision:  {metrics.PositivePrecision:#.##}");
            Console.WriteLine($"*       PositiveRecall:  {metrics.PositiveRecall:#.##}");
            Console.WriteLine($"*       NegativePrecision:  {metrics.NegativePrecision:#.##}");
            Console.WriteLine($"*       NegativeRecall:  {metrics.NegativeRecall:P2}");
            Console.WriteLine($"************************************************************");
        }

        public static void PrintMultiClassClassificationMetrics(string name, MultiClassClassifierMetrics metrics)
        {
            Console.WriteLine($"************************************************************");
            Console.WriteLine($"*    Metrics for {name} multi-class classification model   ");
            Console.WriteLine($"*-----------------------------------------------------------");
            Console.WriteLine($"    AccuracyMacro = {metrics.AccuracyMacro:0.####}, a value between 0 and 1, the closer to 1, the better");
            Console.WriteLine($"    AccuracyMicro = {metrics.AccuracyMicro:0.####}, a value between 0 and 1, the closer to 1, the better");
            Console.WriteLine($"    LogLoss = {metrics.LogLoss:0.####}, the closer to 0, the better");
            Console.WriteLine($"    LogLoss for class 1 = {metrics.PerClassLogLoss[0]:0.####}, the closer to 0, the better");
            Console.WriteLine($"    LogLoss for class 2 = {metrics.PerClassLogLoss[1]:0.####}, the closer to 0, the better");
            Console.WriteLine($"    LogLoss for class 3 = {metrics.PerClassLogLoss[2]:0.####}, the closer to 0, the better");
            Console.WriteLine($"************************************************************");
        }

        public static void PrintRegressionFoldsAverageMetrics(string algorithmName, CrossValidationResult<RegressionMetrics>[] crossValidationResults)
        {
            var L1 = crossValidationResults.Select(r => r.Metrics.L1);
            var L2 = crossValidationResults.Select(r => r.Metrics.L2);
            var RMS = crossValidationResults.Select(r => r.Metrics.L1);
            var lossFunction = crossValidationResults.Select(r => r.Metrics.LossFn);
            var R2 = crossValidationResults.Select(r => r.Metrics.RSquared);

            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for {algorithmName} Regression model      ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       Average L1 Loss:    {L1.Average():0.###} ");
            Console.WriteLine($"*       Average L2 Loss:    {L2.Average():0.###}  ");
            Console.WriteLine($"*       Average RMS:          {RMS.Average():0.###}  ");
            Console.WriteLine($"*       Average Loss Function: {lossFunction.Average():0.###}  ");
            Console.WriteLine($"*       Average R-squared: {R2.Average():0.###}  ");
            Console.WriteLine($"*************************************************************************************************************");
        }

        public static void PrintMulticlassClassificationFoldsAverageMetrics(
                                         string algorithmName,
                                        CrossValidationResult<MultiClassClassifierMetrics>[] crossValResults
                                                                           )
        {
            var metricsInMultipleFolds = crossValResults.Select(r => r.Metrics);

            var microAccuracyValues = metricsInMultipleFolds.Select(m => m.AccuracyMicro);
            var microAccuracyAverage = microAccuracyValues.Average();
            var microAccuraciesStdDeviation = CalculateStandardDeviation(microAccuracyValues);
            var microAccuraciesConfidenceInterval95 = CalculateConfidenceInterval95(microAccuracyValues);

            var macroAccuracyValues = metricsInMultipleFolds.Select(m => m.AccuracyMacro);
            var macroAccuracyAverage = macroAccuracyValues.Average();
            var macroAccuraciesStdDeviation = CalculateStandardDeviation(macroAccuracyValues);
            var macroAccuraciesConfidenceInterval95 = CalculateConfidenceInterval95(macroAccuracyValues);

            var logLossValues = metricsInMultipleFolds.Select(m => m.LogLoss);
            var logLossAverage = logLossValues.Average();
            var logLossStdDeviation = CalculateStandardDeviation(logLossValues);
            var logLossConfidenceInterval95 = CalculateConfidenceInterval95(logLossValues);

            var logLossReductionValues = metricsInMultipleFolds.Select(m => m.LogLossReduction);
            var logLossReductionAverage = logLossReductionValues.Average();
            var logLossReductionStdDeviation = CalculateStandardDeviation(logLossReductionValues);
            var logLossReductionConfidenceInterval95 = CalculateConfidenceInterval95(logLossReductionValues);

            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for {algorithmName} Multi-class Classification model      ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       Average MicroAccuracy:    {microAccuracyAverage:0.###}  - Standard deviation: ({microAccuraciesStdDeviation:#.###})  - Confidence Interval 95%: ({microAccuraciesConfidenceInterval95:#.###})");
            Console.WriteLine($"*       Average MacroAccuracy:    {macroAccuracyAverage:0.###}  - Standard deviation: ({macroAccuraciesStdDeviation:#.###})  - Confidence Interval 95%: ({macroAccuraciesConfidenceInterval95:#.###})");
            Console.WriteLine($"*       Average LogLoss:          {logLossAverage:#.###}  - Standard deviation: ({logLossStdDeviation:#.###})  - Confidence Interval 95%: ({logLossConfidenceInterval95:#.###})");
            Console.WriteLine($"*       Average LogLossReduction: {logLossReductionAverage:#.###}  - Standard deviation: ({logLossReductionStdDeviation:#.###})  - Confidence Interval 95%: ({logLossReductionConfidenceInterval95:#.###})");
            Console.WriteLine($"*************************************************************************************************************");

        }

        public static double CalculateStandardDeviation(IEnumerable<double> values)
        {
            double average = values.Average();
            double sumOfSquaresOfDifferences = values.Select(val => (val - average) * (val - average)).Sum();
            double standardDeviation = Math.Sqrt(sumOfSquaresOfDifferences / (values.Count() - 1));
            return standardDeviation;
        }

        public static double CalculateConfidenceInterval95(IEnumerable<double> values)
        {
            double confidenceInterval95 = 1.96 * CalculateStandardDeviation(values) / Math.Sqrt((values.Count() - 1));
            return confidenceInterval95;
        }

        public static void PrintClusteringMetrics(string name, ClusteringMetrics metrics)
        {
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Metrics for {name} clustering model      ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       AvgMinScore: {metrics.AvgMinScore}");
            Console.WriteLine($"*       DBI is: {metrics.Dbi}");
            Console.WriteLine($"*************************************************");
        }

        public static void ShowDataViewInConsole(MLContext mlContext, IDataView dataView, int numberOfRows = 4)
        {
            string msg = string.Format("Show data in DataView: Showing {0} rows with the columns", numberOfRows.ToString());
            ConsoleWriteHeader(msg);

            var preViewTransformedData = dataView.Preview(maxRows: numberOfRows);

            foreach (var row in preViewTransformedData.RowView)
            {
                var ColumnCollection = row.Values;
                string lineToPrint = "Row--> ";
                foreach (KeyValuePair<string, object> column in ColumnCollection)
                {
                    lineToPrint += $"| {column.Key}:{column.Value}";
                }
                Console.WriteLine(lineToPrint + "\n");
            }
        }

        public static void PeekDataViewInConsole(MLContext mlContext, IDataView dataView, IEstimator<ITransformer> pipeline, int numberOfRows = 4)
        {
            string msg = string.Format("Peek data in DataView: Showing {0} rows with the columns", numberOfRows.ToString());
            ConsoleWriteHeader(msg);

            //https://github.com/dotnet/machinelearning/blob/master/docs/code/MlNetCookBook.md#how-do-i-look-at-the-intermediate-data
            var transformer = pipeline.Fit(dataView);
            var transformedData = transformer.Transform(dataView);

            // 'transformedData' is a 'promise' of data, lazy-loading. call Preview  
            //and iterate through the returned collection from preview.

            var preViewTransformedData = transformedData.Preview(maxRows: numberOfRows);

            foreach (var row in preViewTransformedData.RowView)
            {
                var ColumnCollection = row.Values;
                string lineToPrint = "Row--> ";
                foreach (KeyValuePair<string, object> column in ColumnCollection)
                {
                    lineToPrint += $"| {column.Key}:{column.Value}";
                }
                Console.WriteLine(lineToPrint + "\n");
            }
        }

        public static List<float[]> PeekVectorColumnDataInConsole(MLContext mlContext, string columnName, IDataView dataView, IEstimator<ITransformer> pipeline, int numberOfRows = 4)
        {
            string msg = string.Format("Peek data in DataView: : Show {0} rows with just the '{1}' column", numberOfRows, columnName);
            ConsoleWriteHeader(msg);

            var transformer = pipeline.Fit(dataView);
            var transformedData = transformer.Transform(dataView);

            // Extract the 'Features' column.
            var someColumnData = transformedData.GetColumn<float[]>(mlContext, columnName)
                                                        .Take(numberOfRows).ToList();

            // print to console the peeked rows
            someColumnData.ForEach(row =>
            {
                String concatColumn = String.Empty;
                foreach (float f in row)
                {
                    concatColumn += f.ToString();
                }
                Console.WriteLine(concatColumn);
            });

            return someColumnData;
        }

        public static void ConsoleWriteHeader(params string[] lines)
        {
            var defaultColor = Console.ForegroundColor;
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine(" ");
            foreach (var line in lines)
            {
                Console.WriteLine(line);
            }
            var maxLength = lines.Select(x => x.Length).Max();
            Console.WriteLine(new string('#', maxLength));
            Console.ForegroundColor = defaultColor;
        }

        public static void ConsoleWriterSection(params string[] lines)
        {
            var defaultColor = Console.ForegroundColor;
            Console.ForegroundColor = ConsoleColor.Blue;
            Console.WriteLine(" ");
            foreach (var line in lines)
            {
                Console.WriteLine(line);
            }
            var maxLength = lines.Select(x => x.Length).Max();
            Console.WriteLine(new string('-', maxLength));
            Console.ForegroundColor = defaultColor;
        }

        public static void ConsolePressAnyKey()
        {
            var defaultColor = Console.ForegroundColor;
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine(" ");
            Console.WriteLine("Press any key to finish.");
            Console.ReadKey();
        }

        public static void ConsoleWriteException(params string[] lines)
        {
            var defaultColor = Console.ForegroundColor;
            Console.ForegroundColor = ConsoleColor.Red;
            const string exceptionTitle = "EXCEPTION";
            Console.WriteLine(" ");
            Console.WriteLine(exceptionTitle);
            Console.WriteLine(new string('#', exceptionTitle.Length));
            Console.ForegroundColor = defaultColor;
            foreach (var line in lines)
            {
                Console.WriteLine(line);
            }
        }

        public static void ConsoleWriteWarning(params string[] lines)
        {
            var defaultColor = Console.ForegroundColor;
            Console.ForegroundColor = ConsoleColor.DarkMagenta;
            const string warningTitle = "WARNING";
            Console.WriteLine(" ");
            Console.WriteLine(warningTitle);
            Console.WriteLine(new string('#', warningTitle.Length));
            Console.ForegroundColor = defaultColor;
            foreach (var line in lines)
            {
                Console.WriteLine(line);
            }
        }
    }
}
