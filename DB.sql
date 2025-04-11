-- MySQL dump 10.13  Distrib 8.0.41, for Win64 (x86_64)
--
-- Host: localhost    Database: documentprocessing
-- ------------------------------------------------------
-- Server version	8.0.41

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Dumping data for table `agencydocuments`
--

LOCK TABLES `agencydocuments` WRITE;
/*!40000 ALTER TABLE `agencydocuments` DISABLE KEYS */;
/*!40000 ALTER TABLE `agencydocuments` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Dumping data for table `batches`
--

LOCK TABLES `batches` WRITE;
/*!40000 ALTER TABLE `batches` DISABLE KEYS */;
INSERT INTO `batches` VALUES (45,'Batch_20250411_083151','2025-04-11 12:31:51'),(46,'Batch_20250411_084610','2025-04-11 12:46:10');
/*!40000 ALTER TABLE `batches` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Dumping data for table `checkdocuments`
--

LOCK TABLES `checkdocuments` WRITE;
/*!40000 ALTER TABLE `checkdocuments` DISABLE KEYS */;
INSERT INTO `checkdocuments` VALUES (64,'8705191245','683487045',1730.00),(65,'FLD5539003745','129821120',610.00);
/*!40000 ALTER TABLE `checkdocuments` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Dumping data for table `claimdocuments`
--

LOCK TABLES `claimdocuments` WRITE;
/*!40000 ALTER TABLE `claimdocuments` DISABLE KEYS */;
/*!40000 ALTER TABLE `claimdocuments` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Dumping data for table `coupondocuments`
--

LOCK TABLES `coupondocuments` WRITE;
/*!40000 ALTER TABLE `coupondocuments` DISABLE KEYS */;
/*!40000 ALTER TABLE `coupondocuments` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Dumping data for table `documents`
--

LOCK TABLES `documents` WRITE;
/*!40000 ALTER TABLE `documents` DISABLE KEYS */;
INSERT INTO `documents` VALUES (62,45,5,'processed','2025-04-11 12:32:27','MRTG CHANGE PPWK 4.pdf','{\n    \"filename\": \"MRTG CHANGE PPWK 4.pdf\",\n    \"category\": \"Mortgage\",\n    \"Important_Info\": {\n        \"policynumber\": \"240601\",\n        \"name\": \"CLINTON BARTLETT JR\",\n        \"address\": \"3230 SEA CASTLE DR, CRYSTAL BEACH TX, 77650\"\n    }\n}'),(63,46,4,'exception','2025-04-11 12:46:50','20250320_14788 B701 CK 59326 $574.00.tif','{\"filename\": \"20250320_14788 B701 CK 59326 $574.00.tif\", \"category\": \"Check\", \"Important_Info\": {\"policynumber\": \"8707410446\", \"loannumber\": \"111111\", \"amount\": \"574.00\", \"payee\": \"NGM Insurance Company\", \"payee_address\": \"152 Church Street, Canaan, CT 06018\", \"payee_phone\": \"860-693-5009\"}}'),(64,46,4,'processed','2025-04-11 12:46:50','20250320_15539 B708 CK 2184933 $1730.00.tif','{\n    \"filename\": \"20250320_15539 B708 CK 2184933 $1730.00.tif\",\n    \"category\": \"Check\",\n    \"Important_Info\": {\n        \"policynumber\": \"8705191245\",\n        \"loannumber\": \"683487045\",\n        \"amount\": \"1730.00\"\n    }\n}'),(65,46,4,'processed','2025-04-11 12:46:50','20250320_15539 B708 CK 1886344 $610.00.tif','{\n    \"filename\": \"20250320_15539 B708 CK 1886344 $610.00.tif\",\n    \"category\": \"Check\",\n    \"Important_Info\": {\n        \"policynumber\": \"FLD5539003745\",\n        \"loannumber\": \"129821120\",\n        \"amount\": \"610.00\"\n    }\n}');
/*!40000 ALTER TABLE `documents` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Dumping data for table `documenttypes`
--

LOCK TABLES `documenttypes` WRITE;
/*!40000 ALTER TABLE `documenttypes` DISABLE KEYS */;
INSERT INTO `documenttypes` VALUES (8,'agency'),(4,'check'),(1,'Cheque'),(6,'claim'),(3,'Contract'),(7,'coupons'),(5,'mortgage'),(9,'no money - cancel'),(2,'Supporting Documents'),(10,'unknown');
/*!40000 ALTER TABLE `documenttypes` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Dumping data for table `exceptionresolutions`
--

LOCK TABLES `exceptionresolutions` WRITE;
/*!40000 ALTER TABLE `exceptionresolutions` DISABLE KEYS */;
/*!40000 ALTER TABLE `exceptionresolutions` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Dumping data for table `exceptions`
--

LOCK TABLES `exceptions` WRITE;
/*!40000 ALTER TABLE `exceptions` DISABLE KEYS */;
INSERT INTO `exceptions` VALUES (20,63,'Missing loannumber',0,'2025-04-11 12:46:50','Missing Data');
/*!40000 ALTER TABLE `exceptions` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Dumping data for table `mortgagedocuments`
--

LOCK TABLES `mortgagedocuments` WRITE;
/*!40000 ALTER TABLE `mortgagedocuments` DISABLE KEYS */;
INSERT INTO `mortgagedocuments` VALUES (62,'240601','CLINTON BARTLETT JR','3230 SEA CASTLE DR, CRYSTAL BEACH TX, 77650');
/*!40000 ALTER TABLE `mortgagedocuments` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2025-04-11 10:35:06
