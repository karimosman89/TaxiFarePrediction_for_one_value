using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML;

class Predictions
{
    static void Main(string[] args)
    {
        // File paths
        string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-train.csv");
        string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-test.csv");
        string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");

        // ML Context initialization
        MLContext mlContext = new MLContext(seed: 0);

        // Train the model
        ITransformer model = Train(mlContext, _trainDataPath);

        // Evaluate the model
        Evaluate(mlContext, model, _testDataPath);

        // Predict fare amounts for train and test data and write to files
        PredictFareAndWriteToFile(mlContext, model, _trainDataPath, "train_predicted.csv");
        PredictFareAndWriteToFile(mlContext, model, _testDataPath, "test_predicted.csv");

        Console.WriteLine("Prediction completed and files saved.");
    }

    static ITransformer Train(MLContext mlContext, string dataPath)
    {
        // Load data
        IDataView dataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(dataPath, hasHeader: true, separatorChar: ',');

        // Define data preparation pipeline and regression algorithm
        var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "FareAmount")
            .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "VendorIdEncoded", inputColumnName: "VendorId"))
            .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "RateCodeEncoded", inputColumnName: "RateCode"))
            .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PaymentTypeEncoded", inputColumnName: "PaymentType"))
            .Append(mlContext.Transforms.Concatenate("Features", "VendorIdEncoded", "RateCodeEncoded", "PassengerCount", "TripDistance", "PaymentTypeEncoded"))
            .Append(mlContext.Regression.Trainers.FastTree());

        // Train the model
        var trainedModel = pipeline.Fit(dataView);

        return trainedModel;
    }

    static void Evaluate(MLContext mlContext, ITransformer model, string testDataPath)
    {
        // Load test data
        IDataView testDataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(testDataPath, hasHeader: true, separatorChar: ',');

        // Make predictions
        IDataView predictions = model.Transform(testDataView);

        // Evaluate the model
        var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");

        // Output evaluation metrics
        Console.WriteLine();
        Console.WriteLine($"*************************************************");
        Console.WriteLine($"*       Model quality metrics evaluation         ");
        Console.WriteLine($"*------------------------------------------------");
        Console.WriteLine($"*       RSquared Score:      {metrics.RSquared:0.##}");
        Console.WriteLine($"*       Root Mean Squared Error:      {metrics.RootMeanSquaredError:#.##}");
    }

    static void PredictFareAndWriteToFile(MLContext mlContext, ITransformer model, string dataPath, string outputPath)
    {
        // Load data
        IDataView dataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(dataPath, hasHeader: true, separatorChar: ',');

        // Create prediction engine
        var predictionEngine = mlContext.Model.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(model);

        // Predict fare amounts
        IEnumerable<TaxiTrip> trips = mlContext.Data.CreateEnumerable<TaxiTrip>(dataView, reuseRowObject: false);
        List<TaxiTrip> predictedTrips = new List<TaxiTrip>();

        foreach (var trip in trips)
        {
            var prediction = predictionEngine.Predict(trip);
            trip.PredictedFareAmount = prediction.FareAmount;
            predictedTrips.Add(trip);
        }

        // Write data with predicted fare to CSV
        using (var writer = new StreamWriter(outputPath))
        {
            writer.WriteLine("VendorId,RateCode,PassengerCount,TripTime,TripDistance,PaymentType,FareAmount,PredictedFareAmount");

            foreach (var trip in predictedTrips)
            {
                writer.WriteLine($"{trip.VendorId},{trip.RateCode},{trip.PassengerCount},{trip.TripTime},{trip.TripDistance},{trip.PaymentType},{trip.FareAmount},{trip.PredictedFareAmount}");
            }
        }
    }
}
