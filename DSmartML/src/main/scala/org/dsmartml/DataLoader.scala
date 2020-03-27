//package org.apache.spark.ml.tuning
package org.dsmartml
import org.apache.spark.ml.feature.{MinMaxScaler, StringIndexer, VectorAssembler}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DataTypes, DoubleType, IntegerType, NumericType}
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
  * Class Name: DataLoader
  * Description: this calss responable for loading datasets that used in building the KB ,it contains a list of the datasets name, delimeters and if they has header or not
  * @constructor Create an Object of Data Loader class, to load KB datasets
  * @param spark the used spark session
  * @param ds the id of the dataset (0 up to 76)
  * @param Path the path of the datasets
  * @param logger Object of the logger class (used to log time, exception and result)
  * @param PresistData if i want to persist the dataset or not (default = true)
  * @param Partations if i want to set the parations number manually (default = 0 -> set partation number automatically)
  * @author Ahmed Eissa
  * @version 1.0
  * @Date 22/5/2019
*/
class DataLoader (spark:SparkSession, ds: Int,  Path : String, logger: Logger, PresistData: Boolean = true, Partations: Int = 0){


  /**
    * Array of the datasets used to build our KB initially, with Dataset (array index as the Id of the dataset in the KB, dataset name and Dataset Source URL in the comment)
    */
  val datasets = Array(
    /*0- */"00-covtype.data" ,                  //https://archive.ics.uci.edu/ml/datasets/Covertype
    /*1- */"01-census-income.data" ,            //https://archive.ics.uci.edu/ml/datasets/Census-Income+%28KDD%29
    /*2- */"02-kddcup.data" ,                   //https://archive.ics.uci.edu/ml/datasets/KDD+Cup+1999+Data
    /*3- */"03-poker-hand-testing.data" ,       //https://archive.ics.uci.edu/ml/datasets/Poker+Hand
    /*4- */"04-credit_card_clients.csv" ,       //https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
    /*5- */"05-Sensorless_drive_diagnosis.csv" ,//https://archive.ics.uci.edu/ml/datasets/Dataset%2Bfor%2BSensorless%2BDrive%2BDiagnosis
    /*6- */"06-Statlog_Shuttle.csv" ,           //https://archive.ics.uci.edu/ml/datasets/Statlog+%28Shuttle%29
    /*7- */"07-avila.txt" ,                     //https://archive.ics.uci.edu/ml/datasets/Avila
    /*8- */"08-SUSY.csv" ,                      //https://archive.ics.uci.edu/ml/datasets/SUSY
    /*9- */"09-Skin_NonSkin.txt" ,              //https://archive.ics.uci.edu/ml/datasets/Skin+Segmentation
    /*10 */"10-dota2Train.csv" ,                //https://archive.ics.uci.edu/ml/datasets/Dota2+Games+Results
    /*11 */"11-MoCap_Hand_Postures.csv" ,       //https://archive.ics.uci.edu/ml/datasets/MoCap+Hand+Postures
    /*12 */"12-IDA2016Challenge.csv" ,          //https://archive.ics.uci.edu/ml/datasets/IDA2016Challenge
    /*13*/ "13-adult.csv" ,                     //https://archive.ics.uci.edu/ml/datasets/Adult
    /*14*/ "14-bank-full.csv" ,                 //https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
    /*15*/ "15-train_5xor_128dim.csv",          //train_5xor_128dim.csv"  //https://archive.ics.uci.edu/ml/datasets/Physical+Unclonable+Functions#
    /*16*/ "16-train_6xor_64dim.csv",           //https://archive.ics.uci.edu/ml/datasets/Physical+Unclonable+Functions#
    /*17*/ "17-The_broken_machine.csv",         //https://www.kaggle.com/ivanloginov/the-broken-machine
    /*18*/ "18-sensor.csv",                     //https://www.kaggle.com/nphantawee/pump-sensor-data
    /*19*/ "19-aps_failure_training.csv",       //https://archive.ics.uci.edu/ml/datasets/APS+Failure+at+Scania+Trucks
    /*20*/ "20-HT_Sensor_dataset.csv" ,         //https://archive.ics.uci.edu/ml/datasets/Gas+sensors+for+home+activity+monitoring
    /*21*/ "21-samsung_train.csv" ,             //https://www.kaggle.com/kashnitsky/mlcourse
    /*22*/ "22-flight_delays_train.csv" ,       //https://www.kaggle.com/kashnitsky/mlcourse
    /*23*/ "23-HIGGS.csv" ,                     //https://archive.ics.uci.edu/ml/datasets/HIGGS
    /*24*/ "24-telecom_churn.csv" ,             //https://www.kaggle.com/jpacse/datasets-for-churn-telecom
    /*25*/ "25-winequality-white.csv" ,         //https://www.kaggle.com/kashnitsky/mlcourse
    /*26*/ "26-weatherAUS.csv" ,                //URL:https://www.kaggle.com/jsphyg/weather-dataset-rattle-package
    /*27*/ "27-mitbih_train.csv" ,              //URL:https://www.kaggle.com/shayanfazeli/heartbeat#mitbih_train.csv
    /*28*/ "28-ptbdb.csv" ,                     //URL:https://www.kaggle.com/shayanfazeli/heartbeat#mitbih_train.csv
    /*29*/ "29-loan-default-prediction.csv" ,   //URL:https://www.kaggle.com/roshansharma/loan-default-prediction#train.csv
    /*30*/ "30-cell2celltrain.csv"  ,           //URL:https://www.kaggle.com/jpacse/datasets-for-churn-telecom#cell2celltrain.csv
    /*31*/ "31-benign" ,                        //https://archive.ics.uci.edu/ml/datasets/detection_of_IoT_botnet_attacks_N_BaIoT
    /*32*/ "32-OCRB.csv" ,                      //https://archive.ics.uci.edu/ml/datasets/Character+Font+Images
    /*33*/ "33-DataforPersonActivity.csv" ,     //https://archive.ics.uci.edu/ml/datasets/Localization+Data+for+Person+Activity
    /*34*/ "34-diabetic_data.csv" ,             //https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008
    /*35*/ "35-Connect4.csv"  ,                 //URL:https://archive.ics.uci.edu/ml/datasets/Connect-4
    /*36*/ "36-Nomao.csv" ,                     //URL:https://archive.ics.uci.edu/ml/machine-learning-databases/00227/
    /*37*/ "37-Grammatical_Facial_Expressions.csv" , //https://archive.ics.uci.edu/ml/datasets/Grammatical+Facial+Expressions
    /*38*/ "38-Grammatical_Facial_Expressions_relative_datapoints.csv" , //https://archive.ics.uci.edu/ml/datasets/Grammatical+Facial+Expressions
    /*39*/ "39-Grammatical_Facial_Expressions_conditional_datapoints.csv" ,
    /*40*/ "40-UJIIndoorLoc.csv" ,              //https://archive.ics.uci.edu/ml/datasets/UJIIndoorLoc
    /*41*/ "41-HTRU_2.csv" ,                    //https://archive.ics.uci.edu/ml/datasets/HTRU2
    /*42*/ "42-p53_Mutants.csv" ,               //https://archive.ics.uci.edu/ml/datasets/p53+Mutants
    /*43*/ "43-EEG Eye State.csv" ,             //https://archive.ics.uci.edu/ml/datasets/EEG+Eye+State
    /*44*/ "44-Gas_Sensor_Array_Drift.csv" ,    //https://archive.ics.uci.edu/ml/datasets/Gas+Sensor+Array+Drift+Dataset+at+Different+Concentrations
    /*45*/ "45-Gisette.csv" ,                   //https://archive.ics.uci.edu/ml/datasets/Gisette
    /*46*/ "46-housenumbers.csv" ,              //https://www.kaggle.com/olgabelitskaya/svhn-preproccessed-fragments
    /*47*/ "47-mnist-original.csv",             //https://www.kaggle.com/avnishnish/mnist-original
    /*48*/ "48-A_DeviceMotion_data.csv" ,       //https://www.kaggle.com/malekzadeh/motionsense-dataset#data_subjects_info.csv
    /*49*/ "49-KaggleV2-May-2016.csv" ,         //https://www.kaggle.com/joniarroba/noshowappointments
    /*50*/ "50-FAO.csv" ,                       //https://www.kaggle.com/dorbicycle/world-foodfeed-production
    /*51*/ "51-fashion-mnist_train.csv" ,       //https://www.kaggle.com/zalando-research/fashionmnist
    /*52*/ "52-hmnist_28_28_RGB.csv" ,          //https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000
    /*53*/ "53-creditcard.csv" ,                //https://www.kaggle.com/mlg-ulb/creditcardfraud
    /*54*/ "54-HEPMASS.csv"   ,                  //https://archive.ics.uci.edu/ml/datasets/HEPMASS
    /*55*/ "55-online_shoppers_intention.csv" , //https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset#
    /*56*/ "56-data.csv" , //https://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition
    /*57*/ "57-SmartphoneBasedRecognition.csv" ,  //https://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions
    /*58*/ "58-Crowdsourced_Mapping.csv" , //https://archive.ics.uci.edu/ml/datasets/Crowdsourced+Mapping
    /*59*/ "59-Polish_companies_bankruptcy.csv" , //https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data
    /*60*/ "60-HumanActivityRecognition.csv" , //https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones
    /*61*/ "61-Data_for_UCI_named.csv" , //https://archive.ics.uci.edu/ml/datasets/Electrical+Grid+Stability+Simulated+Data+
    /*62*/ "62-Frogs_MFCCs.csv" , //https://archive.ics.uci.edu/ml/datasets/Anuran+Calls+%28MFCCs%29
    /*63*/ "63-Musk.csv" , //https://archive.ics.uci.edu/ml/datasets/Musk+%28Version+2%29
    /*64*/ "64-turkiye-student_evaluation_generic.csv" , //https://archive.ics.uci.edu/ml/datasets/Turkiye+Student+Evaluation
    /*65*/ "65-dataset_uci.csv" , //https://archive.ics.uci.edu/ml/datasets/Smartphone+Dataset+for+Human+Activity+Recognition+%28HAR%29+in+Ambient+Assisted+Living+%28AAL%29
    /*66*/ "66-Optical_Recognition-of_Handwritten.csv" , //https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits
    /*67*/ "67-page-blocks.csv" , //https://archive.ics.uci.edu/ml/datasets/Page+Blocks+Classification
    /*68*/ "68-sensor_readings_24.csv" , //https://archive.ics.uci.edu/ml/datasets/Wall-Following+Robot+Navigation+Data
    /*69*/ "69-waveform.csv" , //https://archive.ics.uci.edu/ml/datasets/Waveform+Database+Generator+%28Version+2%29
    /*70*/ "70-madelon.csv" , //https://archive.ics.uci.edu/ml/datasets/Madelon
    /*71*/ "71-c2k_data_comma.csv" , //https://archive.ics.uci.edu/ml/datasets/Cargo+2000+Freight+Tracking+and+Tracing
    /*72*/ "72-USPS.csv" , //https://www.kaggle.com/bistaumanga/usps-dataset
    /*73*/  "73-list_attr_celeba.csv" , //https://www.kaggle.com/jessicali9530/celeba-dataset
    /*74*/  "74-pulsar_stars.csv" , //https://www.kaggle.com/pavanraj159/predicting-a-pulsar-star
    /*75*/ "75-ClassifyGestures.csv" , //https://www.kaggle.com/kyr7plus/emg-4/downloads/emg-4.zip/2
    /*76*/ "76-Handwriting.csv" , //https://www.kaggle.com/anupamwadhwa/handwriting-verification
    /*77*/ "KB2.csv" ,
    /*78*/ "78-Seattle_Crime_Data_06-23-2019-4.csv" , //https://www.openml.org/d/41960 ,
    /*79*/ "79-php89ntbG.csv" ,//https://www.openml.org/d/351
    /*80*/ "80-file7b53746cbda2.csv" ,//https://www.openml.org/d/41147
    /*81*/ "81-MiniBooNE.csv" ,  //https://www.openml.org/d/41150
    /*82*/ "82-BayesianNetworkGenerator_optdigits.csv" , //https://www.openml.org/d/123
    /*83*/ "83-BayesianNetworkGenerator_segment.csv" ,// https://www.openml.org/d/130
    /*84*/ "84-BayesianNetworkGenerator_anneal.ORIG_small.csv" ,//https://www.openml.org/d/71
    /*85*/ "85-BayesianNetworkGenerator_kr-vs-kp_small.csv", //https://www.openml.org/d/72
    /*86*/ "86-BayesianNetworkGenerator_letter_small.csv" , //https://www.openml.org/d/74
    /*87*/ "87-BayesianNetworkGenerator_labor_small.csv" , //https://www.openml.org/d/73
    /*88*/ "88-BayesianNetworkGenerator_hypothyroid.csv" , //https://www.openml.org/d/144
    /*89*/ "89-BayesianNetworkGenerator_mfeat-zernike.csv" , //https://www.openml.org/d/118
    /*90*/ "90-BayesianNetworkGenerator_hepatitis.csv" , //https://www.openml.org/d/142
    /*91*/ "91-BayesianNetworkGenerator_mfeat-fourier_small.csv" , //https://www.openml.org/d/78
    /*92*/ "92-CovPokElec.csv" , //https://www.openml.org/d/149
    /*93*/ "93-AirlinesCodrnaAdult.csv" , //https://www.openml.org/d/1240
    /*94*/ "94-Dataset-Unicauca-Version2-87Atts.csv" //https://www.kaggle.com/jsrojas/ip-network-traffic-flows-labeled-with-87-apps

  )

  /**
    * this function load the dataset tacking of consideration (has header or not, what is the separator, rename the label column to "y", convert string to index, remove unwanted column,...)
    * @return Dataframe of the requested dataset
    */
  def getData():DataFrame = {
    //logger.logTime(datasets(ds) + ",")
    val starttime =  new java.util.Date().getTime

    // Read Data
    var rawdata = spark.read.option("header",hasHeader(ds))
    .option("inferSchema","true")
    .option("delimiter", getDelimeter(ds))
    .format("csv")
    .load(Path + datasets(ds))

    //Partation it
    if(Partations > 0 )
    rawdata = rawdata.repartition(Partations)


    //Special cases
    if(ds == 0) {
      rawdata = rawdata.withColumnRenamed("_c54", "y")
    }

    if(ds == 1) {
    rawdata = rawdata.withColumnRenamed("_c41", "y")

    val indexer1 = new StringIndexer().setInputCol("_c1").setOutputCol("_c1_").fit(rawdata)
    val indexer2 = new StringIndexer().setInputCol("_c4").setOutputCol("_c4_").fit(rawdata)
    val indexer3 = new StringIndexer().setInputCol("_c6").setOutputCol("_c6_").fit(rawdata)
    val indexer4 = new StringIndexer().setInputCol("_c7").setOutputCol("_c7_").fit(rawdata)
    val indexer5 = new StringIndexer().setInputCol("_c8").setOutputCol("_c8_").fit(rawdata)
    val indexer6 = new StringIndexer().setInputCol("_c9").setOutputCol("_c9_").fit(rawdata)
    val indexer7 = new StringIndexer().setInputCol("_c10").setOutputCol("_c10_").fit(rawdata)
    val indexer8 = new StringIndexer().setInputCol("_c11").setOutputCol("_c11_").fit(rawdata)
    val indexer9 = new StringIndexer().setInputCol("_c12").setOutputCol("_c12_").fit(rawdata)
    val indexer10 = new StringIndexer().setInputCol("_c13").setOutputCol("_c13_").fit(rawdata)
    val indexer11 = new StringIndexer().setInputCol("_c14").setOutputCol("_c14_").fit(rawdata)
    val indexer12 = new StringIndexer().setInputCol("_c15").setOutputCol("_c15_").fit(rawdata)
    val indexer13 = new StringIndexer().setInputCol("_c19").setOutputCol("_c19_").fit(rawdata)
    val indexer14 = new StringIndexer().setInputCol("_c20").setOutputCol("_c20_").fit(rawdata)
    val indexer15 = new StringIndexer().setInputCol("_c21").setOutputCol("_c21_").fit(rawdata)
    val indexer16 = new StringIndexer().setInputCol("_c22").setOutputCol("_c22_").fit(rawdata)
    val indexer17 = new StringIndexer().setInputCol("_c23").setOutputCol("_c23_").fit(rawdata)
    val indexer18 = new StringIndexer().setInputCol("_c25").setOutputCol("_c25_").fit(rawdata)
    val indexer19 = new StringIndexer().setInputCol("_c26").setOutputCol("_c26_").fit(rawdata)
    val indexer20 = new StringIndexer().setInputCol("_c27").setOutputCol("_c27_").fit(rawdata)
    val indexer21 = new StringIndexer().setInputCol("_c28").setOutputCol("_c28_").fit(rawdata)
    val indexer22 = new StringIndexer().setInputCol("_c29").setOutputCol("_c29_").fit(rawdata)
    val indexer23 = new StringIndexer().setInputCol("_c31").setOutputCol("_c31_").fit(rawdata)
    val indexer24 = new StringIndexer().setInputCol("_c32").setOutputCol("_c32_").fit(rawdata)
    val indexer25 = new StringIndexer().setInputCol("_c33").setOutputCol("_c33_").fit(rawdata)
    val indexer26 = new StringIndexer().setInputCol("_c34").setOutputCol("_c34_").fit(rawdata)
    val indexer27 = new StringIndexer().setInputCol("_c35").setOutputCol("_c35_").fit(rawdata)
    val indexer28 = new StringIndexer().setInputCol("_c37").setOutputCol("_c37_").fit(rawdata)


    rawdata = indexer1.transform(rawdata).drop("_c1")
    rawdata = indexer2.transform(rawdata).drop("_c4")
    rawdata = indexer3.transform(rawdata).drop("_c6")
    rawdata = indexer4.transform(rawdata).drop("_c7")
    rawdata = indexer5.transform(rawdata).drop("_c8")
    rawdata = indexer6.transform(rawdata).drop("_c9")
    rawdata = indexer7.transform(rawdata).drop("_c10")
    rawdata = indexer8.transform(rawdata).drop("_c11")
    rawdata = indexer9.transform(rawdata).drop("_c12")
    rawdata = indexer10.transform(rawdata).drop("_c13")
    rawdata = indexer11.transform(rawdata).drop("_c14")
    rawdata = indexer12.transform(rawdata).drop("_c15")
    rawdata = indexer13.transform(rawdata).drop("_c19")
    rawdata = indexer14.transform(rawdata).drop("_c20")
    rawdata = indexer15.transform(rawdata).drop("_c21")
    rawdata = indexer16.transform(rawdata).drop("_c22")
    rawdata = indexer17.transform(rawdata).drop("_c23")
    rawdata = indexer18.transform(rawdata).drop("_c25")
    rawdata = indexer19.transform(rawdata).drop("_c26")
    rawdata = indexer20.transform(rawdata).drop("_c27")
    rawdata = indexer21.transform(rawdata).drop("_c28")
    rawdata = indexer22.transform(rawdata).drop("_c29")
    rawdata = indexer23.transform(rawdata).drop("_c31")
    rawdata = indexer24.transform(rawdata).drop("_c32")
    rawdata = indexer25.transform(rawdata).drop("_c33")
    rawdata = indexer26.transform(rawdata).drop("_c34")
    rawdata = indexer27.transform(rawdata).drop("_c35")
    rawdata = indexer28.transform(rawdata) .drop("_c37")

    import org.apache.spark.sql.functions._
    rawdata = rawdata.withColumn("y", when(col("y") === " - 50000.", 0).otherwise("1").cast(org.apache.spark.sql.types.DataTypes.FloatType));

  }

    if(ds == 2) {

    val indexer1 = new StringIndexer().setInputCol("_c1").setOutputCol("_c1_").fit(rawdata)
    val indexer2 = new StringIndexer().setInputCol("_c2").setOutputCol("_c2_").fit(rawdata)
    val indexer3 = new StringIndexer().setInputCol("_c3").setOutputCol("_c3_").fit(rawdata)
    val indexer4 = new StringIndexer().setInputCol("_c41").setOutputCol("y").fit(rawdata)

    rawdata = indexer1.transform(rawdata).drop("_c1")
    rawdata = indexer2.transform(rawdata).drop("_c2")
    rawdata = indexer3.transform(rawdata).drop("_c3")
    rawdata = indexer4.transform(rawdata).drop("_c41")

  }

    if(ds == 3) {

    rawdata = rawdata.withColumnRenamed("_c10", "y")
  }

    if(ds == 6) {
    rawdata = rawdata.withColumnRenamed("_c9" , "y")

  }

    if(ds == 7) {
    rawdata = rawdata.withColumnRenamed("_c10" , "y")

  }

    if(ds == 8) {
    rawdata = rawdata.withColumn("y" , col("_c0").cast(IntegerType))
    rawdata = rawdata.drop("_c0")

  }

    if(ds == 9) {

    rawdata = rawdata.withColumn("y" ,  when(col("_c3").equalTo(2), 1).otherwise( lit(0)) )
    rawdata = rawdata.drop("_c3")
    //rawdata = rawdata.withColumnRenamed("_c3", "y")

  }

    if(ds == 10) {

    rawdata = rawdata.withColumn("y" ,  when(col("_c0").equalTo(-1), 0).otherwise( lit(1)) )
    rawdata = rawdata.drop("_c0")
    //rawdata = rawdata.withColumnRenamed("_c3", "y")

  }

    if(ds == 11) {

    rawdata = rawdata.withColumn("y" , ( col("Class") - 1).cast(IntegerType))

    rawdata = rawdata.drop("Class")

    rawdata.schema.filter(c => c.name != "y").foreach(scol => {

    if(! scol.dataType.isInstanceOf[NumericType])
  {
    rawdata = rawdata.withColumn(scol.name + "_1" ,  when(col(scol.name).equalTo("?"), null).otherwise( scol.name).cast(IntegerType) )
    rawdata = rawdata.drop(scol.name)
  }
  })
    //rawdata = rawdata.withColumnRenamed("_c3", "y"var del


  }

    if(ds == 12) {
    rawdata = rawdata.withColumnRenamed("class", "y")

  }

    if(ds == 13) {
    rawdata = rawdata.withColumn( "y" ,  when(col("_c14" ).equalTo(" <=50K"), 0).otherwise( lit(1)).cast(IntegerType) )
    rawdata = rawdata.drop("_c14")


    rawdata.schema.filter(c => c.name != "y").foreach(scol => {

    //if(! scol.dataType.isInstanceOf[NumericType])
    //{
    rawdata = rawdata.withColumn(scol.name + "_1" ,  when(col(scol.name).equalTo("?"), null).otherwise( col(scol.name) ))
    rawdata = rawdata.drop(scol.name)
    //}
  })


    val indexer1 = new StringIndexer().setInputCol("_c1_1").setOutputCol("c1")
    rawdata = indexer1.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("_c1_1")

    val indexer3 = new StringIndexer().setInputCol("_c3_1").setOutputCol("c3")
    rawdata = indexer3.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("_c3_1")

    val indexer5 = new StringIndexer().setInputCol("_c5_1").setOutputCol("c5")
    rawdata = indexer5.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("_c5_1")


    val indexer6 = new StringIndexer().setInputCol("_c6_1").setOutputCol("c6")
    rawdata = indexer6.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("_c6_1")

    val indexer7 = new StringIndexer().setInputCol("_c7_1").setOutputCol("c7")
    rawdata = indexer7.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("_c7_1")


    val indexer8 = new StringIndexer().setInputCol("_c8_1").setOutputCol("c8")
    rawdata = indexer8.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("_c8_1")

    val indexer9 = new StringIndexer().setInputCol("_c9_1").setOutputCol("c9")
    rawdata = indexer9.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("_c9_1")

    val indexer13 = new StringIndexer().setInputCol("_c13_1").setOutputCol("c13")
    rawdata = indexer13.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("_c13_1")

  }

    if(ds == 14) {
    rawdata = rawdata.withColumn( "y_1" ,  when(col("y" ).equalTo("yes"), 1).otherwise( lit(0)).cast(IntegerType) )
    rawdata = rawdata.drop("y")
    rawdata = rawdata.withColumnRenamed( "y_1" ,"y")


    rawdata.schema.filter(c => c.name != "y").foreach(scol => {

    //if(! scol.dataType.isInstanceOf[NumericType])
    //{
    rawdata = rawdata.withColumn(scol.name + "_1" ,  when(col(scol.name).equalTo("none"), null).otherwise( col(scol.name) ))
    rawdata = rawdata.drop(scol.name)
    //}
  })


    val indexer1 = new StringIndexer().setInputCol("job_1").setOutputCol("job")
    rawdata = indexer1.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("job_1")

    val indexer3 = new StringIndexer().setInputCol("marital_1").setOutputCol("marital")
    rawdata = indexer3.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("marital_1")

    val indexer5 = new StringIndexer().setInputCol("education_1").setOutputCol("education")
    rawdata = indexer5.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("education_1")


    val indexer6 = new StringIndexer().setInputCol("default_1").setOutputCol("default")
    rawdata = indexer6.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("default_1")

    val indexer7 = new StringIndexer().setInputCol("housing_1").setOutputCol("housing")
    rawdata = indexer7.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("housing_1")


    val indexer8 = new StringIndexer().setInputCol("loan_1").setOutputCol("loan")
    rawdata = indexer8.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("loan_1")

    val indexer9 = new StringIndexer().setInputCol("contact_1").setOutputCol("contact")
    rawdata = indexer9.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("contact_1")

    val indexer13 = new StringIndexer().setInputCol("month_1").setOutputCol("month")
    rawdata = indexer13.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("month_1")

    val indexer14 = new StringIndexer().setInputCol("poutcome_1").setOutputCol("poutcome")
    rawdata = indexer14.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("poutcome_1")

  }

    if(ds == 15) {

    rawdata = rawdata.withColumn( "y" ,  when(col("_c128" ).equalTo(-1), 0).otherwise( lit(1)).cast(IntegerType) )
    rawdata = rawdata.drop("_c128")
  }

    if(ds == 16) {
    rawdata = rawdata.withColumn( "y" ,  when(col("_c64" ).equalTo(-1), 0).otherwise( lit(1)).cast(IntegerType) )
    rawdata = rawdata.drop("_c64")
  }

    if(ds == 17) {
    rawdata = rawdata.withColumnRenamed("x" , "y")
  }

    if(ds == 18) {
    rawdata = rawdata.drop("sensor_15")
    val indexer1 = new StringIndexer().setInputCol("machine_status").setOutputCol("y1")
    rawdata = indexer1.fit(rawdata).transform(rawdata)
    rawdata = rawdata.withColumn("y" , col("y1").cast(DataTypes.IntegerType))
    rawdata = rawdata.drop("machine_status").drop("y1").drop("_c0").drop("timestamp")
  }

    if(ds == 19) {

    val indexer1 = new StringIndexer().setInputCol("class").setOutputCol("y").setHandleInvalid("keep")
    rawdata = indexer1.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("class")

  }

    if(ds == 20) {
    var rawdata1 = rawdata.as("DS") //.select( "_c0" , "_c2", "_c4", "_c6", "_c8", "_c10", "_c12", "_c14", "_c16", "_c18", "_c20", "_c22", "_c24")

    var rawdata2 = spark.read.option("header",false)
    .option("inferSchema","true")
    .option("delimiter", ",")
    .format("csv")
    .load(Path + datasets(ds) + "_1")
    .as("MD")
    var label = "y"

    rawdata2 = rawdata2.withColumnRenamed("_c2" ,"class_")
    rawdata = rawdata1.join(rawdata2 , col("DS._c0") === col("MD._c0") , "inner")
    rawdata = rawdata.select("DS._c1", "DS._c2", "DS._c3", "DS._c4", "DS._c5", "DS._c6", "DS._c7", "DS._c8", "DS._c9", "DS._c10", "DS._c11", "class_")

    //rawdata = rawdata.withColumn( "y" ,  when(col("MD.class" ).equalTo(-1), 0).otherwise( lit(1)).cast(IntegerType) )
    //rawdata = rawdata.drop("_c128")

    val indexer1 = new StringIndexer().setInputCol("class_").setOutputCol("y1")
    rawdata = indexer1.fit(rawdata).transform(rawdata)
    rawdata = rawdata.withColumn("y" , col("y1").cast(DataTypes.IntegerType))
    rawdata = rawdata.drop("class_").drop("y1")


  }

    if(ds == 21) {
    /*
      var rawdata2 = spark.read.option("header",false)
        .option("inferSchema","true")
        .option("delimiter", ",")
        .format("csv")
        .load(Path + datasets(ds) + "_1")

      rawdata = rawdata.union(rawdata2)
      */
    //rawdata = rawdata.withColumnRenamed("_c561" ,"y")
    rawdata = rawdata.withColumn("y" , ( col("_c561") - 1).cast(IntegerType))
    rawdata = rawdata.drop("_c561") .drop("_c0")
  }

    if(ds == 22) {
    val indexer1 = new StringIndexer().setInputCol("UniqueCarrier").setOutputCol("UniqueCarrier_")
    rawdata = indexer1.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("UniqueCarrier")

    val indexer2 = new StringIndexer().setInputCol("Origin").setOutputCol("Origin_")
    rawdata = indexer2.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("Origin")

    val indexer3 = new StringIndexer().setInputCol("Dest").setOutputCol("Dest_")
    rawdata = indexer3.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("Dest")
  }

    if(ds == 23) {

      rawdata = rawdata.toDF( "y" , "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
        "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18" , "C19", "C20", "C21", "C22", "C23"
        , "C24", "C25", "C26", "C27", "C28")

      rawdata = rawdata.withColumn("y", rawdata("y").cast(org.apache.spark.sql.types.DataTypes.FloatType))
      rawdata = rawdata.withColumn("C1", rawdata("C1").cast(org.apache.spark.sql.types.DataTypes.FloatType))
      rawdata = rawdata.withColumn("C2", rawdata("C2").cast(org.apache.spark.sql.types.DataTypes.FloatType))
      rawdata = rawdata.withColumn("C3", rawdata("C3").cast(org.apache.spark.sql.types.DataTypes.FloatType))
      rawdata = rawdata.withColumn("C4", rawdata("C4").cast(org.apache.spark.sql.types.DataTypes.FloatType))
      rawdata = rawdata.withColumn("C5", rawdata("C5").cast(org.apache.spark.sql.types.DataTypes.FloatType))
      rawdata = rawdata.withColumn("C6", rawdata("C6").cast(org.apache.spark.sql.types.DataTypes.FloatType))
      rawdata = rawdata.withColumn("C7", rawdata("C7").cast(org.apache.spark.sql.types.DataTypes.FloatType))
      rawdata = rawdata.withColumn("C8", rawdata("C8").cast(org.apache.spark.sql.types.DataTypes.FloatType))
      rawdata = rawdata.withColumn("C9", rawdata("C9").cast(org.apache.spark.sql.types.DataTypes.FloatType))
      rawdata = rawdata.withColumn("C10", rawdata("C10").cast(org.apache.spark.sql.types.DataTypes.FloatType))
      rawdata = rawdata.withColumn("C11", rawdata("C11").cast(org.apache.spark.sql.types.DataTypes.FloatType))
      rawdata = rawdata.withColumn("C12", rawdata("C12").cast(org.apache.spark.sql.types.DataTypes.FloatType))
      rawdata = rawdata.withColumn("C13", rawdata("C13").cast(org.apache.spark.sql.types.DataTypes.FloatType))
      rawdata = rawdata.withColumn("C14", rawdata("C14").cast(org.apache.spark.sql.types.DataTypes.FloatType))
      rawdata = rawdata.withColumn("C15", rawdata("C15").cast(org.apache.spark.sql.types.DataTypes.FloatType))
      rawdata = rawdata.withColumn("C16", rawdata("C16").cast(org.apache.spark.sql.types.DataTypes.FloatType))
      rawdata = rawdata.withColumn("C17", rawdata("C17").cast(org.apache.spark.sql.types.DataTypes.FloatType))
      rawdata = rawdata.withColumn("C18", rawdata("C18").cast(org.apache.spark.sql.types.DataTypes.FloatType))
      rawdata = rawdata.withColumn("C19", rawdata("C19").cast(org.apache.spark.sql.types.DataTypes.FloatType))
      rawdata = rawdata.withColumn("C20", rawdata("C20").cast(org.apache.spark.sql.types.DataTypes.FloatType))
      rawdata = rawdata.withColumn("C21", rawdata("C21").cast(org.apache.spark.sql.types.DataTypes.FloatType))
      rawdata = rawdata.withColumn("C22", rawdata("C22").cast(org.apache.spark.sql.types.DataTypes.FloatType))
      rawdata = rawdata.withColumn("C23", rawdata("C23").cast(org.apache.spark.sql.types.DataTypes.FloatType))
      rawdata = rawdata.withColumn("C24", rawdata("C24").cast(org.apache.spark.sql.types.DataTypes.FloatType))
      rawdata = rawdata.withColumn("C25", rawdata("C25").cast(org.apache.spark.sql.types.DataTypes.FloatType))
      rawdata = rawdata.withColumn("C26", rawdata("C26").cast(org.apache.spark.sql.types.DataTypes.FloatType))
      rawdata = rawdata.withColumn("C27", rawdata("C27").cast(org.apache.spark.sql.types.DataTypes.FloatType))
      rawdata = rawdata.withColumn("C28", rawdata("C28").cast(org.apache.spark.sql.types.DataTypes.FloatType))

    }

    if(ds == 24) {

    val indexer1 = new StringIndexer().setInputCol("State").setOutputCol("State_")
    rawdata = indexer1.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("State")

    rawdata = rawdata.withColumnRenamed("Churn","y")


  }

    if(ds == 25) {
    rawdata = rawdata.withColumnRenamed("quality","y")
  }

    if(ds == 26) {
    rawdata = rawdata.drop("Date")

    val indexer1 = new StringIndexer().setInputCol("Location").setOutputCol("Location_").setHandleInvalid("keep")
    rawdata = indexer1.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("Location")

    val indexer2 = new StringIndexer().setInputCol("WindGustDir").setOutputCol("WindGustDir_").setHandleInvalid("keep")
    rawdata = indexer2.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("WindGustDir")

    val indexer3 = new StringIndexer().setInputCol("WindDir9am").setOutputCol("WindDir9am_").setHandleInvalid("keep")
    rawdata = indexer3.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("WindDir9am")

    val indexer4 = new StringIndexer().setInputCol("WindDir3pm").setOutputCol("WindDir3pm_").setHandleInvalid("keep")
    rawdata = indexer4.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("WindDir3pm")

    val indexer5 = new StringIndexer().setInputCol("RainToday").setOutputCol("RainToday_").setHandleInvalid("keep")
    rawdata = indexer5.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("RainToday")

    val indexer6 = new StringIndexer().setInputCol("RainTomorrow").setOutputCol("y")
    rawdata = indexer6.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("RainTomorrow")


  }

    if(ds == 27) {

    var rawdata2 = spark.read.option("header",false)
    .option("inferSchema","true")
    .option("delimiter", ",")
    .format("csv")
    .load(Path + datasets(ds) + "_1")

    rawdata = rawdata.union(rawdata2)
    rawdata = rawdata.withColumnRenamed("_c187" ,"y")
    //rawdata = rawdata.withColumn("y" , ( col("_c561") - 1).cast(IntegerType))
    //rawdata = rawdata.drop("_c561") .drop("_c0").randomSplit(Array(0.001 , 0.999))(0)
  }

    if(ds == 28) {
    rawdata = rawdata.withColumnRenamed("_c187" ,"y")
  }

    if(ds == 29) {
    rawdata = rawdata.drop("UniqueID").drop("CREDIT.HISTORY.LENGTH").drop("AVERAGE.ACCT.AGE").drop("Date.of.Birth").drop("DisbursalDate").drop("Employee_code_ID")
    rawdata = rawdata.withColumnRenamed("Employment.Type","EmploymentType")
    rawdata = rawdata.withColumnRenamed("PERFORM_CNS.SCORE.DESCRIPTION","x")
    rawdata = rawdata.withColumnRenamed("PERFORM_CNS.SCORE","PERFORM_CNS_SCORE")
    rawdata = rawdata.withColumnRenamed("PRI.NO.OF.ACCTS","PRI_NO_OF_ACCTS")
    rawdata = rawdata.withColumnRenamed("PRI.ACTIVE.ACCTS","PRI_ACTIVE_ACCTS")
    rawdata = rawdata.withColumnRenamed("PRI.OVERDUE.ACCTS","PRI_OVERDUE_ACCTS")
    rawdata = rawdata.withColumnRenamed("PRI.CURRENT.BALANCE","PRI_CURRENT_BALANCE")
    rawdata = rawdata.withColumnRenamed("PRI.SANCTIONED.AMOUNT","PRI_SANCTIONED_AMOUNT")
    rawdata = rawdata.withColumnRenamed("PRI.DISBURSED.AMOUNT","PRI_DISBURSED_AMOUNT")
    rawdata = rawdata.withColumnRenamed("SEC.NO.OF.ACCTS","SEC_NO_OF_ACCTS")
    rawdata = rawdata.withColumnRenamed("SEC.ACTIVE.ACCTS","SEC_ACTIVE_ACCTS")
    rawdata = rawdata.withColumnRenamed("SEC.OVERDUE.ACCTS","SEC_OVERDUE_ACCTS")
    rawdata = rawdata.withColumnRenamed("SEC.CURRENT.BALANCE","SEC_CURRENT_BALANCE")
    rawdata = rawdata.withColumnRenamed("SEC.SANCTIONED.AMOUNT","SEC_SANCTIONED_AMOUNT")
    rawdata = rawdata.withColumnRenamed("SEC.DISBURSED.AMOUNT","SEC_DISBURSED_AMOUNT")
    rawdata = rawdata.withColumnRenamed("PRIMARY.INSTAL.AMT","PRIMARY_INSTAL_AMT")
    rawdata = rawdata.withColumnRenamed("SEC.INSTAL.AMT","SEC_INSTAL_AMT")
    rawdata = rawdata.withColumnRenamed("NEW.ACCTS.IN.LAST.SIX.MONTHS","NEW_ACCTS_IN_LAST_SIX_MONTHS")
    rawdata = rawdata.withColumnRenamed("DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS","DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS")
    rawdata = rawdata.withColumnRenamed("AVERAGE.ACCT","AVERAGE_ACCT")
    rawdata = rawdata.withColumnRenamed("CREDIT.HISTORY","CREDIT_HISTORY")
    rawdata = rawdata.withColumnRenamed("NO.OF_INQUIRIES","NO_OF_INQUIRIES")




    val indexer1 = new StringIndexer().setInputCol("EmploymentType").setOutputCol("mployment_Type").setHandleInvalid("keep")
    rawdata = indexer1.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("EmploymentType")

    val indexer2 = new StringIndexer().setInputCol("x").setOutputCol("PERFORM_CNS_SCORE_DESCRIPTION").setHandleInvalid("keep")
    rawdata = indexer2.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("x")


  }

    if(ds == 30) {
    rawdata = rawdata.drop("CustomerID").drop("HandsetPrice")

    val indexer1 = new StringIndexer().setInputCol("Churn").setOutputCol("y").setHandleInvalid("keep")
    rawdata = indexer1.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("Churn")

    val indexer2 = new StringIndexer().setInputCol("ServiceArea").setOutputCol("ServiceArea_").setHandleInvalid("keep")
    rawdata = indexer2.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("ServiceArea")

    val indexer3 = new StringIndexer().setInputCol("ChildrenInHH").setOutputCol("ChildrenInHH_").setHandleInvalid("keep")
    rawdata = indexer3.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("ChildrenInHH")

    val indexer4 = new StringIndexer().setInputCol("HandsetRefurbished").setOutputCol("HandsetRefurbished_").setHandleInvalid("keep")
    rawdata = indexer4.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("HandsetRefurbished")

    val indexer5 = new StringIndexer().setInputCol("HandsetWebCapable").setOutputCol("HandsetWebCapable_").setHandleInvalid("keep")
    rawdata = indexer5.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("HandsetWebCapable")

    val indexer6 = new StringIndexer().setInputCol("TruckOwner").setOutputCol("TruckOwner_").setHandleInvalid("keep")
    rawdata = indexer6.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("TruckOwner")

    val indexer7 = new StringIndexer().setInputCol("RVOwner").setOutputCol("RVOwner_").setHandleInvalid("keep")
    rawdata = indexer7.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("RVOwner")

    val indexer8 = new StringIndexer().setInputCol("Homeownership").setOutputCol("Homeownership_").setHandleInvalid("keep")
    rawdata = indexer8.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("Homeownership")

    val indexer9 = new StringIndexer().setInputCol("BuysViaMailOrder").setOutputCol("BuysViaMailOrder_").setHandleInvalid("keep")
    rawdata = indexer9.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("BuysViaMailOrder")

    val indexer10 = new StringIndexer().setInputCol("RespondsToMailOffers").setOutputCol("RespondsToMailOffers_").setHandleInvalid("keep")
    rawdata = indexer10.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("RespondsToMailOffers")

    val indexer11 = new StringIndexer().setInputCol("OptOutMailings").setOutputCol("OptOutMailings_").setHandleInvalid("keep")
    rawdata = indexer11.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("OptOutMailings")

    val indexer12 = new StringIndexer().setInputCol("NonUSTravel").setOutputCol("NonUSTravel_").setHandleInvalid("keep")
    rawdata = indexer12.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("NonUSTravel")

    val indexer13 = new StringIndexer().setInputCol("OwnsComputer").setOutputCol("OwnsComputer_").setHandleInvalid("keep")
    rawdata = indexer13.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("OwnsComputer")

    val indexer14 = new StringIndexer().setInputCol("HasCreditCard").setOutputCol("HasCreditCard_").setHandleInvalid("keep")
    rawdata = indexer14.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("HasCreditCard")

    val indexer15 = new StringIndexer().setInputCol("NewCellphoneUser").setOutputCol("NewCellphoneUser_").setHandleInvalid("keep")
    rawdata = indexer15.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("NewCellphoneUser")

    val indexer16 = new StringIndexer().setInputCol("NotNewCellphoneUser").setOutputCol("NotNewCellphoneUser_").setHandleInvalid("keep")
    rawdata = indexer16.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("NotNewCellphoneUser")

    val indexer17 = new StringIndexer().setInputCol("OwnsMotorcycle").setOutputCol("OwnsMotorcycle_").setHandleInvalid("keep")
    rawdata = indexer17.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("OwnsMotorcycle")

    val indexer18 = new StringIndexer().setInputCol("MadeCallToRetentionTeam").setOutputCol("MadeCallToRetentionTeam_").setHandleInvalid("keep")
    rawdata = indexer18.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("MadeCallToRetentionTeam")

    val indexer19 = new StringIndexer().setInputCol("CreditRating").setOutputCol("CreditRating_").setHandleInvalid("keep")
    rawdata = indexer19.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("CreditRating")

    val indexer20 = new StringIndexer().setInputCol("PrizmCode").setOutputCol("PrizmCode_").setHandleInvalid("keep")
    rawdata = indexer20.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("PrizmCode")

    val indexer21 = new StringIndexer().setInputCol("Occupation").setOutputCol("Occupation_").setHandleInvalid("keep")
    rawdata = indexer21.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("Occupation")

    val indexer22 = new StringIndexer().setInputCol("MaritalStatus").setOutputCol("MaritalStatus_").setHandleInvalid("keep")
    rawdata = indexer22.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("MaritalStatus")
  }

    if(ds == 31) {
    //benign - 0
    rawdata = rawdata.withColumn( "y" , lit(0))

    //ack - 1
    var rawdata1 = spark.read.option("header",hasHeader(ds))
    .option("inferSchema","true")
    .option("delimiter", ",")
    .format("csv")
    .load(Path + "31-ack")
    rawdata1 = rawdata1.withColumn( "y" , lit(1))

    //scan - 2
    var rawdata2 = spark.read.option("header",hasHeader(ds))
    .option("inferSchema","true")
    .option("delimiter", ",")
    .format("csv")
    .load(Path + "31-scan")
    rawdata2 = rawdata2.withColumn( "y" , lit(2))

    //syn- 3
    var rawdata3 = spark.read.option("header",hasHeader(ds))
    .option("inferSchema","true")
    .option("delimiter", ",")
    .format("csv")
    .load(Path + "31-syn")
    rawdata3 = rawdata2.withColumn( "y" , lit(3))

    //udp- 4
    var rawdata4 = spark.read.option("header",hasHeader(ds))
    .option("inferSchema","true")
    .option("delimiter", ",")
    .format("csv")
    .load(Path + "31-udp")
    rawdata4 = rawdata2.withColumn( "y" , lit(4))

    rawdata = rawdata.union(rawdata1)
    rawdata = rawdata.union(rawdata2)
    rawdata = rawdata.union(rawdata3)
    rawdata = rawdata.union(rawdata4)

    rawdata.schema.foreach(col => {
    rawdata = rawdata.withColumnRenamed(col.name , col.name.replace("." , "_"))
  })


  }

    if(ds == 32) {

    //32-SEGOE.csv
    var rawdata1 = spark.read.option("header",hasHeader(ds))
    .option("inferSchema","true")
    .option("delimiter", ",")
    .format("csv")
    .load(Path + "32-SEGOE.csv")

    //32-HANDPRINT.csv
    var rawdata2 = spark.read.option("header",hasHeader(ds))
    .option("inferSchema","true")
    .option("delimiter", ",")
    .format("csv")
    .load(Path + "32-HANDPRINT.csv")

    //32-OCRA.csv
    var rawdata3 = spark.read.option("header",hasHeader(ds))
    .option("inferSchema","true")
    .option("delimiter", ",")
    .format("csv")
    .load(Path + "32-OCRA.csv")

    //32-CREDITCARD.csv
    var rawdata4 = spark.read.option("header",hasHeader(ds))
    .option("inferSchema","true")
    .option("delimiter", ",")
    .format("csv")
    .load(Path + "32-CREDITCARD.csv")

    //32-ARIAL.csv
    var rawdata5 = spark.read.option("header",hasHeader(ds))
    .option("inferSchema","true")
    .option("delimiter", ",")
    .format("csv")
    .load(Path + "32-ARIAL.csv")

    //32-E13B.csv
    var rawdata6 = spark.read.option("header",hasHeader(ds))
    .option("inferSchema","true")
    .option("delimiter", ",")
    .format("csv")
    .load(Path + "32-E13B.csv")

    //32-CALIBRI.csv
    var rawdata7 = spark.read.option("header",hasHeader(ds))
    .option("inferSchema","true")
    .option("delimiter", ",")
    .format("csv")
    .load(Path + "32-CALIBRI.csv")

    //32-SITKA.csv
    var rawdata8 = spark.read.option("header",hasHeader(ds))
    .option("inferSchema","true")
    .option("delimiter", ",")
    .format("csv")
    .load(Path + "32-SITKA.csv")

    //32-FRANKLIN.csv
    var rawdata9 = spark.read.option("header",hasHeader(ds))
    .option("inferSchema","true")
    .option("delimiter", ",")
    .format("csv")
    .load(Path + "32-FRANKLIN.csv")

    //32-FRANKLIN.csv
    var rawdata10 = spark.read.option("header",hasHeader(ds))
    .option("inferSchema","true")
    .option("delimiter", ",")
    .format("csv")
    .load(Path + "32-LUCIDA.csv")


    rawdata = rawdata.union(rawdata1)
    rawdata = rawdata.union(rawdata2)
    rawdata = rawdata.union(rawdata3)
    rawdata = rawdata.union(rawdata4)
    rawdata = rawdata.union(rawdata5)
    rawdata = rawdata.union(rawdata6)
    rawdata = rawdata.union(rawdata7)
    rawdata = rawdata.union(rawdata8)
    rawdata = rawdata.union(rawdata9)
    rawdata = rawdata.union(rawdata10)


    val indexer1 = new StringIndexer().setInputCol("font").setOutputCol("y").setHandleInvalid("keep")
    rawdata = indexer1.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("font").drop("fontVariant")

  }

    if(ds == 33) {

    val indexer1 = new StringIndexer().setInputCol("_c7").setOutputCol("y").setHandleInvalid("keep")
    rawdata = indexer1.fit(rawdata).transform(rawdata)

    val indexer2 = new StringIndexer().setInputCol("_c1").setOutputCol("c1").setHandleInvalid("keep")
    rawdata = indexer2.fit(rawdata).transform(rawdata)

    val indexer3 = new StringIndexer().setInputCol("_c0").setOutputCol("c0").setHandleInvalid("keep")
    rawdata = indexer3.fit(rawdata).transform(rawdata)


    rawdata = rawdata.drop("_c7").drop("_c1").drop("_c0").drop("_c3")
  }

    if(ds == 34) {


    val indexer1 = new StringIndexer().setInputCol("readmitted").setOutputCol("y").setHandleInvalid("keep")
    rawdata = indexer1.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("readmitted")

    val indexer2 = new StringIndexer().setInputCol("diag_1").setOutputCol("diag_1_").setHandleInvalid("keep")
    rawdata = indexer2.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("diag_1")

    val indexer3 = new StringIndexer().setInputCol("diag_2").setOutputCol("diag_2_").setHandleInvalid("keep")
    rawdata = indexer3.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("diag_2")

    val indexer4 = new StringIndexer().setInputCol("diag_3").setOutputCol("diag_3_").setHandleInvalid("keep")
    rawdata = indexer4.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("diag_3")

    val indexer5 = new StringIndexer().setInputCol("medical_specialty").setOutputCol("medical_specialty_").setHandleInvalid("keep")
    rawdata = indexer5.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("medical_specialty")

    val indexer6 = new StringIndexer().setInputCol("payer_code").setOutputCol("payer_code_").setHandleInvalid("keep")
    rawdata = indexer6.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("payer_code")

    val indexer7 = new StringIndexer().setInputCol("weight").setOutputCol("weight_").setHandleInvalid("keep")
    rawdata = indexer7.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("weight")

    val indexer8 = new StringIndexer().setInputCol("age").setOutputCol("age_").setHandleInvalid("keep")
    rawdata = indexer8.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("age")

    val indexer9 = new StringIndexer().setInputCol("gender").setOutputCol("gender_").setHandleInvalid("keep")
    rawdata = indexer9.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("gender")

    val indexer10 = new StringIndexer().setInputCol("race").setOutputCol("race_").setHandleInvalid("keep")
    rawdata = indexer10.fit(rawdata).transform(rawdata)
    rawdata = rawdata.drop("race")

    rawdata = rawdata.drop("patient_nbr").drop("encounter_id")
  }

    if(ds == 35) {
    rawdata = rawdata.withColumnRenamed("_c42","y")
  }

    if(ds == 36) {

    rawdata = rawdata.withColumnRenamed("_c118","y")
    //rawdata.select("y").distinct().collect().foreach(x => println(x))
  }

    if(ds == 37) {
    rawdata = rawdata.drop("0")
  }

    if(ds == 38) {
    rawdata = rawdata.drop("0")
  }

    if(ds == 39) {
    rawdata = rawdata.drop("0")
  }

    if(ds == 40) {
    rawdata = rawdata.withColumnRenamed("BUILDINGID" , "y")
    rawdata = rawdata.drop("SPACEID").drop("USERID").drop("PHONEID").drop("TIMESTAMP")
  }

    if(ds == 41) {
    rawdata = rawdata.withColumnRenamed("_c8" , "y")
  }

    if(ds == 42) {
    rawdata = rawdata.withColumnRenamed("_c5408" , "y")
  }

    if(ds == 43) {
    rawdata = rawdata.withColumnRenamed("_c14" , "y")
  }

    if(ds == 44) {
    rawdata = rawdata.withColumnRenamed("_c0" , "y")
  }

    if(ds == 45) {
    rawdata = rawdata.withColumnRenamed("_c0" , "y")
  }

    if(ds == 46) {
    rawdata = rawdata.withColumnRenamed("y4" , "y")
    rawdata = rawdata.drop("y0").drop("y1").drop("y2").drop("y3").drop("filename")
  }

    if(ds == 48) {
    rawdata = rawdata.drop("Unnamed: 0")
    rawdata = rawdata.withColumnRenamed("attitude.pitch" , "attitude_pitch")
    rawdata = rawdata.withColumnRenamed("attitude.roll" , "attitude_roll")
    rawdata = rawdata.withColumnRenamed("attitude.yaw" , "attitude_yaw")
    rawdata = rawdata.withColumnRenamed("gravity.x" , "gravity_x")
    rawdata = rawdata.withColumnRenamed("gravity.y" , "gravity_y")
    rawdata = rawdata.withColumnRenamed("gravity.z" , "gravity_z")
    rawdata = rawdata.withColumnRenamed("rotationRate.x" , "rotationRate_x")
    rawdata = rawdata.withColumnRenamed("rotationRate.y" , "rotationRate_y")
    rawdata = rawdata.withColumnRenamed("rotationRate.z" , "rotationRate_z")
    rawdata = rawdata.withColumnRenamed("userAcceleration.x" , "userAcceleration_x")
    rawdata = rawdata.withColumnRenamed("userAcceleration.y" , "userAcceleration_y")
    rawdata = rawdata.withColumnRenamed("userAcceleration.z" , "userAcceleration_z")

  }

    if(ds == 49) {
        rawdata = rawdata.drop("ScheduledDay").drop("AppointmentDay")

        val indexer1 = new StringIndexer().setInputCol("Neighbourhood").setOutputCol("Neighbourhood_").setHandleInvalid("keep")
        rawdata = indexer1.fit(rawdata).transform(rawdata)
        rawdata = rawdata.drop("Neighbourhood")


      }

    if(ds == 54) {
      rawdata = rawdata.toDF("y" , "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
        "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18" , "C19", "C20", "C21", "C22", "C23"
        , "C24", "C25", "C26", "C27", "C28")

      rawdata = rawdata.withColumn("y", rawdata("y").cast(org.apache.spark.sql.types.DataTypes.FloatType))
      rawdata = rawdata.withColumn("C1", rawdata("C1").cast(org.apache.spark.sql.types.DataTypes.FloatType))
      rawdata = rawdata.withColumn("C2", rawdata("C2").cast(org.apache.spark.sql.types.DataTypes.FloatType))
      rawdata = rawdata.withColumn("C3", rawdata("C3").cast(org.apache.spark.sql.types.DataTypes.FloatType))
      rawdata = rawdata.withColumn("C4", rawdata("C4").cast(org.apache.spark.sql.types.DataTypes.FloatType))
      rawdata = rawdata.withColumn("C5", rawdata("C5").cast(org.apache.spark.sql.types.DataTypes.FloatType))
      rawdata = rawdata.withColumn("C6", rawdata("C6").cast(org.apache.spark.sql.types.DataTypes.FloatType))
      rawdata = rawdata.withColumn("C7", rawdata("C7").cast(org.apache.spark.sql.types.DataTypes.FloatType))
      rawdata = rawdata.withColumn("C8", rawdata("C8").cast(org.apache.spark.sql.types.DataTypes.FloatType))
      rawdata = rawdata.withColumn("C9", rawdata("C9").cast(org.apache.spark.sql.types.DataTypes.FloatType))
      rawdata = rawdata.withColumn("C10", rawdata("C10").cast(org.apache.spark.sql.types.DataTypes.FloatType))
      rawdata = rawdata.withColumn("C11", rawdata("C11").cast(org.apache.spark.sql.types.DataTypes.FloatType))
      rawdata = rawdata.withColumn("C12", rawdata("C12").cast(org.apache.spark.sql.types.DataTypes.FloatType))
      rawdata = rawdata.withColumn("C13", rawdata("C13").cast(org.apache.spark.sql.types.DataTypes.FloatType))
      rawdata = rawdata.withColumn("C14", rawdata("C14").cast(org.apache.spark.sql.types.DataTypes.FloatType))
      rawdata = rawdata.withColumn("C15", rawdata("C15").cast(org.apache.spark.sql.types.DataTypes.FloatType))
      rawdata = rawdata.withColumn("C16", rawdata("C16").cast(org.apache.spark.sql.types.DataTypes.FloatType))
      rawdata = rawdata.withColumn("C17", rawdata("C17").cast(org.apache.spark.sql.types.DataTypes.FloatType))
      rawdata = rawdata.withColumn("C18", rawdata("C18").cast(org.apache.spark.sql.types.DataTypes.FloatType))
      rawdata = rawdata.withColumn("C19", rawdata("C19").cast(org.apache.spark.sql.types.DataTypes.FloatType))
      rawdata = rawdata.withColumn("C20", rawdata("C20").cast(org.apache.spark.sql.types.DataTypes.FloatType))
      rawdata = rawdata.withColumn("C21", rawdata("C21").cast(org.apache.spark.sql.types.DataTypes.FloatType))
      rawdata = rawdata.withColumn("C22", rawdata("C22").cast(org.apache.spark.sql.types.DataTypes.FloatType))
      rawdata = rawdata.withColumn("C23", rawdata("C23").cast(org.apache.spark.sql.types.DataTypes.FloatType))
      rawdata = rawdata.withColumn("C24", rawdata("C24").cast(org.apache.spark.sql.types.DataTypes.FloatType))
      rawdata = rawdata.withColumn("C25", rawdata("C25").cast(org.apache.spark.sql.types.DataTypes.FloatType))
      rawdata = rawdata.withColumn("C26", rawdata("C26").cast(org.apache.spark.sql.types.DataTypes.FloatType))
      rawdata = rawdata.withColumn("C27", rawdata("C27").cast(org.apache.spark.sql.types.DataTypes.FloatType))
      rawdata = rawdata.withColumn("C28", rawdata("C28").cast(org.apache.spark.sql.types.DataTypes.FloatType))

    }

    if(ds == 55) {
      val indexer1 = new StringIndexer().setInputCol("Month").setOutputCol("Month_").setHandleInvalid("keep")
      rawdata = indexer1.fit(rawdata).transform(rawdata)
      rawdata = rawdata.drop("Month")
          //.drop("Administrative_Duration").drop("Administrative").drop("Informational_Duration").drop("Informational")
          //.drop("ProductRelated_Duration").drop("ProductRelated").drop("BounceRates").drop("ExitRates").drop("PageValues")
          //.drop("SpecialDay").drop("Month").drop("OperatingSystems").drop("Browser").drop("Region")
          //.drop("TrafficType").drop("VisitorType").drop("Weekend")
          .drop("Month_")
    }

    if(ds == 57) {
      rawdata = rawdata.withColumnRenamed("_c561","y")
    }

    if(ds == 58) {
      val indexer1 = new StringIndexer().setInputCol("class").setOutputCol("y").setHandleInvalid("keep")
      rawdata = indexer1.fit(rawdata).transform(rawdata)
      rawdata = rawdata.drop("class")
    }

    if(ds == 59) {
      rawdata = rawdata.withColumnRenamed("_c64","y")
    }

    if(ds == 60) {
      rawdata = rawdata.withColumnRenamed("_c561","y")
    }

    if(ds == 62) {
      val indexer1 = new StringIndexer().setInputCol("Species").setOutputCol("y").setHandleInvalid("keep")
      rawdata = indexer1.fit(rawdata).transform(rawdata)

      val indexer2 = new StringIndexer().setInputCol("Family").setOutputCol("Family_").setHandleInvalid("keep")
      rawdata = indexer2.fit(rawdata).transform(rawdata)

      rawdata = rawdata.drop("Species")
      rawdata = rawdata.drop("Family")
    }

    if(ds == 64) {
      rawdata = rawdata.withColumnRenamed("class","y")
    }

    if(ds == 65) {
      rawdata = rawdata.withColumnRenamed("_c561","y")
    }

    if(ds == 67) {
      rawdata = rawdata.withColumnRenamed("_c10","y")
    }

    if(ds == 77) {
      val indexer1 = new StringIndexer().setInputCol("BestAlgorithm").setOutputCol("y").setHandleInvalid("keep")
      rawdata = indexer1.fit(rawdata).transform(rawdata)
      rawdata = rawdata.drop("BestAlgorithm")
    }

    if(ds == 78){
      rawdata = rawdata.drop("Report_Number")
      rawdata = rawdata.withColumn("_Occurred_Time" , col("Occurred_Time").cast(IntegerType))
      rawdata = rawdata.withColumn("_Reported_Time" , col("Reported_Time").cast(IntegerType))
      rawdata = rawdata.drop("Occurred_Time").drop("Reported_Time")

      val indexer1 = new StringIndexer().setInputCol("Crime_Subcategory").setOutputCol("_Crime_Subcategory").fit(rawdata)
      rawdata = indexer1.transform(rawdata).drop("Crime_Subcategory")

      val indexer2 = new StringIndexer().setInputCol("Primary_Offense_Description").setOutputCol("_Primary_Offense_Description").fit(rawdata)
      rawdata = indexer2.transform(rawdata).drop("Primary_Offense_Description")

      val indexer3 = new StringIndexer().setInputCol("Precinct").setOutputCol("_Precinct").fit(rawdata)
      rawdata = indexer3.transform(rawdata).drop("Precinct")

      val indexer4 = new StringIndexer().setInputCol("Sector").setOutputCol("_Sector").fit(rawdata)
      rawdata = indexer4.transform(rawdata).drop("Sector")

      val indexer5 = new StringIndexer().setInputCol("Beat").setOutputCol("_Beat").fit(rawdata)
      rawdata = indexer5.transform(rawdata).drop("Beat")

      val indexer6 = new StringIndexer().setInputCol("Neighborhood").setOutputCol("y").fit(rawdata)
      rawdata = indexer6.transform(rawdata).drop("Neighborhood")
    }

    if(ds == 79){

    }

    if(ds == 80){

      for ( c <- rawdata.columns){
        if( c != "class"){
          rawdata = rawdata.withColumn( c , col(c).cast(DoubleType))
        }
      }
      val map = rawdata.columns.map((_, "0")).toMap
      rawdata = rawdata.na.fill(map).withColumnRenamed("class" , "y")

    }

    if(ds == 81){
      rawdata = rawdata.withColumn("y" ,  when( col("signal").equalTo("True") , lit(1)).otherwise( lit(0) ))
      rawdata = rawdata.drop("signal")
    }

    if(ds == 82){
      for ( c <- rawdata.columns){
        if( c != "class"){
          //print("convert column:" + c + " to be:" +  "c_" + c )
          val indexer = new StringIndexer().setInputCol(c).setOutputCol("c_" + c).fit(rawdata)
          rawdata = indexer.transform(rawdata).drop(c)
          //println(" --- Done")
        }
      }
      rawdata = rawdata.withColumnRenamed( "class" , "y")
    }

    if(ds == 83){
      for ( c <- rawdata.columns){
        //if( c != "class"){
        //print("convert column:" + c + " to be:" +  "c_" + c )
        val indexer = new StringIndexer().setInputCol(c).setOutputCol("c_" + c).fit(rawdata)
        rawdata = indexer.transform(rawdata).drop(c)
        //println(" --- Done")
        // }
      }
      rawdata = rawdata.withColumnRenamed( "c_class" , "y")
    }

    if(ds == 84){
      for ( c <- rawdata.columns){
        val indexer = new StringIndexer().setInputCol(c).setOutputCol("c_" + c).fit(rawdata)
        rawdata = indexer.transform(rawdata).drop(c)
      }
      rawdata = rawdata.withColumnRenamed( "c_class" , "y")
    }

    if(ds == 85){
      for ( c <- rawdata.columns){
        //print("convert column:" + c + " to be:" +  "c_" + c )
        val indexer = new StringIndexer().setInputCol(c).setOutputCol("c_" + c).fit(rawdata)
        rawdata = indexer.transform(rawdata).drop(c)
      }
      rawdata = rawdata.withColumnRenamed( "c_class" , "y")
    }

    if(ds == 86){
      for ( c <- rawdata.columns){
        val indexer = new StringIndexer().setInputCol(c).setOutputCol("c_" + c).fit(rawdata)
        rawdata = indexer.transform(rawdata).drop(c)
      }
      rawdata = rawdata.withColumnRenamed( "c_class" , "y")
    }

    if(ds == 87){
      for ( c <- rawdata.columns){
        val indexer = new StringIndexer().setInputCol(c).setOutputCol("c_" + c).fit(rawdata)
        rawdata = indexer.transform(rawdata).drop(c)
      }
      rawdata = rawdata.withColumnRenamed( "c_class" , "y")
    }

    if(ds == 88){
      for ( c <- rawdata.columns){
        val indexer = new StringIndexer().setInputCol(c).setOutputCol("c_" + c).fit(rawdata)
        rawdata = indexer.transform(rawdata).drop(c)
      }
      rawdata = rawdata.withColumnRenamed( "c_Class" , "y")
    }

    if(ds == 89){
      for ( c <- rawdata.columns){
        val indexer = new StringIndexer().setInputCol(c).setOutputCol("c_" + c).fit(rawdata)
        rawdata = indexer.transform(rawdata).drop(c)
      }
      rawdata = rawdata.withColumnRenamed( "c_Class" , "y")
    }

    if(ds == 90){
      for ( c <- rawdata.columns){
        val indexer = new StringIndexer().setInputCol(c).setOutputCol("c_" + c).fit(rawdata)
        rawdata = indexer.transform(rawdata).drop(c)
      }
      rawdata = rawdata.withColumnRenamed( "c_Class" , "y")
    }

    if(ds == 91){
      for ( c <- rawdata.columns){
        val indexer = new StringIndexer().setInputCol(c).setOutputCol("c_" + c).fit(rawdata)
        rawdata = indexer.transform(rawdata).drop(c)
      }
      rawdata = rawdata.withColumnRenamed( "c_Class" , "y")
    }

    if(ds == 92){
      rawdata = rawdata.withColumnRenamed( "class" , "y")
    }

    if(ds == 93){
      val indexer1 = new StringIndexer().setInputCol("Airline").setOutputCol("c_Airline").fit(rawdata)
      rawdata = indexer1.transform(rawdata).drop("Airline")

      val indexer2 = new StringIndexer().setInputCol("AirportFrom").setOutputCol("c_AirportFrom").fit(rawdata)
      rawdata = indexer2.transform(rawdata).drop("AirportFrom")

      val indexer3 = new StringIndexer().setInputCol("AirportTo").setOutputCol("c_AirportTo").fit(rawdata)
      rawdata = indexer3.transform(rawdata).drop("AirportTo")

      val indexer4 = new StringIndexer().setInputCol("workclass").setOutputCol("c_workclass").fit(rawdata)
      rawdata = indexer4.transform(rawdata).drop("workclass")

      val indexer5 = new StringIndexer().setInputCol("education").setOutputCol("c_education").fit(rawdata)
      rawdata = indexer5.transform(rawdata).drop("education")

      val indexer6 = new StringIndexer().setInputCol("marital-status").setOutputCol("c_marital-status").fit(rawdata)
      rawdata = indexer6.transform(rawdata).drop("marital-status")

      val indexer7 = new StringIndexer().setInputCol("occupation").setOutputCol("c_occupation").fit(rawdata)
      rawdata = indexer7.transform(rawdata).drop("occupation")

      val indexer8 = new StringIndexer().setInputCol("relationship").setOutputCol("c_relationship").fit(rawdata)
      rawdata = indexer8.transform(rawdata).drop("relationship")

      val indexer9 = new StringIndexer().setInputCol("race").setOutputCol("c_race").fit(rawdata)
      rawdata = indexer9.transform(rawdata).drop("race")

      val indexer10 = new StringIndexer().setInputCol("sex").setOutputCol("c_sex").fit(rawdata)
      rawdata = indexer10.transform(rawdata).drop("sex")

      val indexer11 = new StringIndexer().setInputCol("native-country").setOutputCol("c_native-country").fit(rawdata)
      rawdata = indexer11.transform(rawdata).drop("native-country")

      rawdata = rawdata.withColumnRenamed( "Delay" , "y")
    }

    if(ds == 94){
      rawdata = rawdata.drop("Flow.ID").drop("Source.IP").drop("Destination.IP").drop("Timestamp").drop("label").drop("L7Protocol")
      rawdata = rawdata.withColumn("ProtocolName" ,  when( col("ProtocolName").notEqual("GOOGLE") && col("ProtocolName").notEqual("HTTP") && col("ProtocolName").notEqual("HTTP_PROXY") , "Other").otherwise( col("ProtocolName") ))
      val indexer = new StringIndexer().setInputCol("ProtocolName").setOutputCol("y").fit(rawdata)
      rawdata = indexer.transform(rawdata).drop("ProtocolName")
      var TargetCol = "y"
      rawdata = rawdata.drop("search_date_pacific").drop("class_id").withColumnRenamed("apply", TargetCol )

      for ( c <- rawdata.columns){
        rawdata = rawdata.withColumnRenamed ( c , c.replace("." , ""))
      }
      //rawdata = rawdata.withColumnRenamed( "Delay" , "y")
    }
    //Persis it
    if(PresistData)
      rawdata.persist()


    val endtime =  new java.util.Date().getTime
    val TotalTime = endtime - starttime
    //logger.logTime( "- Loading Data:" + TotalTime/1000.0 + " sec.\n")
    //println( "Loading Data:" + TotalTime + ",")

    rawdata

  }

  /**
    * this function tell us Which KB dataset has Header
    * @param i the id of the dataset
    * @return boolean (true = has header row , false = with no header row)
    */
  def hasHeader(i:Int):Boolean = {
    var result =  false
    var arr = Array(4,5,11,12,14,17,18,19,22,24,25,26,29,30,31,32,34,37,38,39,40,46,47,48,49,50,51,52,53,55,56,58,61,62,63,64,66,68,69,70,71,72,73,74,75,76,77,
      78,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94)
    if ( arr.contains(i))
      result = true

    result
  }

  /**
    * this function tell us what is the Dataset Delimiter
    * @param i the id of the dataset
    * @return the delimeter as string
    */
  def getDelimeter(i:Int):String = {
      var delimiter = ","
      if (ds == 9)
          delimiter = "\t"
      if(ds ==14)
          delimiter = ";"

      delimiter
      }


}

object DataLoader{

  /**
    * This Function Create Assemble Vector of the feature columns
    * @param df input dataframe
    * @param featureCol output features vector column name
    * @param TargetCol target column name
    * @return proccessed Dataframe
    */
  def convertDFtoVecAssembly(df:DataFrame, featureCol:String, TargetCol:String): DataFrame =
  {
    val featurecolumns = df.columns.filter(c => c != TargetCol)
    val assembler = new VectorAssembler()
      .setInputCols(featurecolumns)
      .setOutputCol(featureCol)
    var mydataset = assembler.transform(df.na.drop).select(TargetCol, featureCol)
    return mydataset
  }

  def convertDFtoVecAssembly_WithoutLabel(df:DataFrame, featureCol:String): DataFrame =
  {
    val featurecolumns = df.columns
    val assembler = new VectorAssembler()
      .setInputCols(featurecolumns)
      .setOutputCol(featureCol)
    var mydataset = assembler.transform(df.na.drop).select(featureCol)
    return mydataset
  }

  /**
    * This Function Scale dataframe features using Min-Max Scaler
    * @param df input dataframe
    * @param featureCol output features vector column name
    * @param TargetCol target column name
    * @return proccessed Dataframe
    */
  def ScaleDF (df:DataFrame , featureCol:String  , TargetCol:String): DataFrame =
  {
    //Scale Dataframe (Min & Max) Scalling
    val scalerMinMax = new MinMaxScaler()
      .setInputCol(featureCol)
      .setOutputCol(featureCol + "_" )
    val scalerMinMaxModel = scalerMinMax.fit(df)
    var df_MinMaxScaled = scalerMinMaxModel.transform(df).select(TargetCol, featureCol + "_" )
    df_MinMaxScaled = df_MinMaxScaled.drop(featureCol).withColumnRenamed(featureCol + "_" , featureCol )
    return df_MinMaxScaled

  }

}
