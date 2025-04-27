package com.sjtu.KGStudy.abnormalDataRec;

import com.sjtu.KGStudy.abnormalDataRec.utils.*;

import com.sjtu.KGStudy.utils.*;
import org.apache.jena.query.*;
import org.apache.jena.rdf.model.Model;
import org.apache.jena.rdf.model.Resource;
import org.apache.jena.util.FileUtils;
import org.topbraid.jenax.util.JenaUtil;
import org.topbraid.shacl.util.ModelPrinter;
import org.topbraid.shacl.validation.ValidationUtil;

import java.io.*;
import java.util.HashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static com.sjtu.KGStudy.main.ShaclReasonTest.analysisMonitorAnomalyReasonReport;
import static com.sjtu.KGStudy.main.ShaclReasonTest.getMonitorData;

public class ServiceMainProgram {
    public static void main(String[] args) {
        // 0. 创建CSM本体和异常诊断的形状图
        Model dataModel = JenaUtil.createMemoryModel();
        Model shapesModelMonitorAnomaly = JenaUtil.createMemoryModel();

        String csmFilepath = "G:/KG_study/reason_main/src/main/resources/myDataGraph/CableStructureMonitor1221.ttl";

        // dataModel就是被检测的模型
        try {
            InputStream inDataModel = new FileInputStream(csmFilepath);
            dataModel.read(inDataModel, "urn:dummy", FileUtils.langTurtle);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        shapesModelMonitorAnomaly.read(ServiceMainProgram.class.getResourceAsStream("/abnormalDataRecServiceData/shapesGraph/monitorAnomaly.ttl"), "urn:dummy", FileUtils.langTurtle);

        boolean check_option = true;
        // int target_row_index = 308;
        int target_row_index = 51;
        while (check_option) {
            // 1. 获取监测数据
            // 1.1 访问数据库获取监测数据
            String tableName = "case1testdata";
            MonitoringDataProcessor processor = new MonitoringDataProcessor(tableName);
            String jsonFileName = processor.processRowByIndex(target_row_index);

            // 1.2 将当前时刻的监测数据和历史监测数据读入
            String monitorDataPath = "G:/KG_study/reason_main/src/main/resources/abnormalDataRecServiceData/" + jsonFileName + ".json";
            Map<String, Object[]> monitorDataMap = getMonitorData(monitorDataPath);
            String historicalMonitorDataPath = "G:/KG_study/reason_main/src/main/resources/abnormalDataRecServiceData/last_50_rows.csv";
            Map<String, float[]> historicalMonitorDataMap = new HashMap<>();
            try {
                historicalMonitorDataMap = CSVUtilsUse.readCSV(historicalMonitorDataPath);
            } catch (IOException e) {
                e.printStackTrace();
            }

            // 依次判断各个观测行为是否正常
            for (Map.Entry<String, Object[]> entry : monitorDataMap.entrySet()){
                String key = entry.getKey();
                // key = "cableTension16";
                // key = "environmentalTemperature8";

                String monitorTime = (String) monitorDataMap.get(key)[0];

                double monitorVaule = (Double) monitorDataMap.get(key)[1];

                // 定义正则表达式
                Pattern pattern = Pattern.compile("(\\D+)(\\d+)");
                Matcher matcher = pattern.matcher(key);

                String textPart = "";
                String numberPart = "";
                if (matcher.matches()) {
                    textPart = matcher.group(1);
                    numberPart = matcher.group(2);
                } else {
                    System.out.println("No match found.");
                    System.exit(1);
                }

                // 2. 更新CSM本体和形状图中的数值
                String subjectStr = "inst:" + textPart + "Obs" + numberPart;  // 确定传感器观测行为的RDF节点

                // 2.1 动态更新数据模型
                RDFModelUpdateTools.updateMonitorTime(dataModel, subjectStr, monitorTime);
                RDFModelUpdateTools.updateMonitorData(dataModel, subjectStr, monitorVaule);
                /*QuerySPARQLUtils.queryMonitorData(dataModel, predicate);*/

                // 2.2 获取阈值并动态更新形状图
                float[] thresholdList = CSVUtilsUse.calculateThresholds(historicalMonitorDataMap.get(key));  // 通过key获取历史数据，并利用3σ准则计算上下界
                RDFModelUpdateTools.updateShapesGraphMonitorData(shapesModelMonitorAnomaly, subjectStr, thresholdList[0], thresholdList[1]);

                // 3. 基于SHACL的监测数据异常诊断
                Resource reportMonitorAnomaly = ValidationUtil.validateModel(dataModel, shapesModelMonitorAnomaly, true);
                /*System.out.println(ModelPrinter.get().print(reportMonitorAnomaly.getModel()));*/

                Object[] reasonResults = analysisMonitorAnomalyReasonReport(reportMonitorAnomaly);
                String focusNode = (String) reasonResults[0];
                boolean labelOption = (boolean) reasonResults[1];  // labelOption为true表示该观察产生了异常数据

                // 3.1 诊断当前观测是否异常
                boolean currentObsStatus = !labelOption; // currentObsStatus为true表示观测正常，为false表示观测异常
                DelSHACLReasonResultMonitorData.labelObservationStatus(dataModel, subjectStr, currentObsStatus);
                if (currentObsStatus){  //如果不是异常观测，就必须利用SPARQL查到所校验的传感器，这样才能保证后续程序正常执行
                    focusNode = QuerySPARQLUtils.getTargetSensorURI(dataModel, subjectStr);
                }

                if (key.contains("environmentalTemperature")) {
                    // 对于环境温度传感器，需要根据当前观测情况调整异常历史记录的index
                    DelHistoricalAbnormalLabel.labelHistoricalObservationStatus(dataModel, subjectStr, currentObsStatus);
                    if (labelOption){// 此外，对判断为异常数据的当前环境观测，直接将数据库数据置为-10000.0
                        UpdateEnvTempCurrentAbnormalObs updateOp = new UpdateEnvTempCurrentAbnormalObs(tableName);
                        updateOp.updateCellValue(key, -10000.0, jsonFileName, true);
                    }
                }

                /*try {
                    int abnormalIndex = DelHistoricalAbnormalLabel.getHistoricalObservationStatus(dataModel, subjectStr);
                    System.out.println(subjectStr + "当前的历史异常标签是：" + abnormalIndex);
                } catch (CustomHisAbnormalIndexException e) {
                    e.printStackTrace();
                }*/

                // break;
            }

            // 4. 确定数据异常的模式
            // 4.1 通过查询与推理确定重构模式
            /*ObsStatusChecking.queryTest(dataModel);*/
            Model queryResultModel = ObsStatusChecking.getQueryResult(dataModel);
            Model shapesModelRecMode = JenaUtil.createMemoryModel();
            String recModeSGFilepath = "G:/KG_study/reason_main/src/main/resources/abnormalDataRecServiceData/shapesGraph/cableTensionReconstructionMode.ttl";
            try {
                InputStream inRecSG = new FileInputStream(recModeSGFilepath);
                shapesModelRecMode.read(inRecSG, "urn:dummy", FileUtils.langTurtle);
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }
            Resource reportRecMode = ValidationUtil.validateModel(queryResultModel, shapesModelRecMode, true);  // 基于SHACL的重构模式选择
            System.out.println(ModelPrinter.get().print(reportRecMode.getModel()));

            // 4.2 解析验证报告并对重构模式进行编码，然后发送到python重构模块的关键文件夹下
            RecModeReportDel.encodeRecModeInfo(reportRecMode, dataModel);

            String targetFolderPath = "G:/KG_study/reason_main/src/main/resources/abnormalDataRecServiceData";
            String targetFileName = "recovered_cable_force.json";
            // 5. 选择对应的代理模型完成数据重构，在python中执行
            try {
                PauseJavaUtils.waitForDataReconstruction(targetFolderPath, targetFileName);
            } catch (IOException e) {
                e.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            // 6. 解码更新结果更新数据库
            UpdateEnvTempCurrentAbnormalObs updateOp = new UpdateEnvTempCurrentAbnormalObs(tableName);
            updateOp.updateRecoveredData(targetFolderPath+"/"+targetFileName,  jsonFileName);  // jsonFileName指的是时间戳，也即当前诊断时刻的时间戳

            // 清空目标文件夹
            File folder = new File(targetFolderPath);
            if (folder.exists() && folder.isDirectory()) {
                // 删除文件夹内的非文件夹文件
                FileCleanUtils.deleteFilesInFolder(folder);
            } else {
                System.out.println("指定路径不存在或不是文件夹。");
            }

            target_row_index ++;
            // check_option=false;
            if (target_row_index == 337) {
                check_option = false;
            }
        }
    }
}
