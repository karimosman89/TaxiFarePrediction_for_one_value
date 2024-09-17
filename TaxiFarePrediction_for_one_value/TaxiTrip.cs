using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;

public class TaxiTrip
{
    [LoadColumn(0)]
    public string? VendorId;

    [LoadColumn(1)]
    public string? RateCode;

    [LoadColumn(2)]
    public float PassengerCount;

    [LoadColumn(3)]
    public float TripTime;

    [LoadColumn(4)]
    public float TripDistance;

    [LoadColumn(5)]
    public string? PaymentType;

    [LoadColumn(6)]
    public float FareAmount;

    [LoadColumn(7)]
    public float PredictedFareAmount { get; set; }
}

public class TaxiTripFarePrediction
{
    [ColumnName("Score")]
    public float FareAmount;
}