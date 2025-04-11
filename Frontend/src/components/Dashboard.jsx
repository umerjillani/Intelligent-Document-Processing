import React, { useState, useEffect } from 'react';
import { Row, Col, Card, Statistic, Table, Tag, Upload, message, Spin } from 'antd';
import { Pie } from 'react-chartjs-2';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';
import { InboxOutlined } from '@ant-design/icons';
import axios from 'axios';

ChartJS.register(ArcElement, Tooltip, Legend);
const { Dragger } = Upload;

const Dashboard = () => {
  // State for storing API data
  const [dashboardData, setDashboardData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Upload props
  const uploadProps = {
    name: 'file',
    multiple: true,
    action: '/api/upload', // Update with your actual upload endpoint
    onChange(info) {
      const { status } = info.file;
      if (status !== 'uploading') {
        console.log(info.file, info.fileList);
      }
      if (status === 'done') {
        message.success(`${info.file.name} file uploaded successfully.`);
      } else if (status === 'error') {
        message.error(`${info.file.name} file upload failed.`);
      }
    },
  };

  // Document type data for pie chart - this could come from API in a real implementation
  const documentTypeData = {
    labels: ['Claims', 'Agency Services', 'Mortgage Changes', 'Coupons'],
    datasets: [{
      data: [42, 28, 18, 12],
      backgroundColor: ['#1890ff', '#ff4d4f', '#52c41a', '#faad14']
    }]
  };

  // Batch table columns
  const batchColumns = [
    {
      title: 'Batch ID',
      dataIndex: 'batch_id',
      key: 'batch_id',
    },
    {
      title: 'Documents',
      dataIndex: 'documents',
      key: 'documents',
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status) => {
        const color = {
          'Complete': 'green',
          'Processing': 'blue',
          'Exceptions': 'red'
        }[status] || 'default';
        return <Tag color={color}>{status}</Tag>;
      }
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_, record) => <a href={`/batches/${record.id}`}>View</a>
    }
  ];

  // Exceptions table columns
  const exceptionsColumns = [
    {
      title: 'Document ID',
      dataIndex: 'doc_id',
      key: 'doc_id',
    },
    {
      title: 'Type',
      dataIndex: 'type',
      key: 'type',
    },
    {
      title: 'Issue',
      dataIndex: 'issue',
      key: 'issue',
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_, record) => <a href={`/documents/${record.doc_id}`}>Resolve</a>
    }
  ];

  // Fetch dashboard data from API
  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        setLoading(true);
        const response = await axios.get('http://localhost:3000/api/dashboard');
        console.log(response.data)
        setDashboardData(response.data);
        setLoading(false);
      } catch (err) {
        setError('Failed to fetch dashboard data');
        console.error('Error fetching dashboard data:', err);
        setLoading(false);
      }
    };

    fetchDashboardData();
  }, []);

  // Process batch data from array format to object format for table
  const processBatchData = (batchesArray) => {
    if (!batchesArray || !Array.isArray(batchesArray)) return [];
    
    return batchesArray.map(batch => ({
      id: batch[0],
      batch_id: batch[1],
      documents: batch[2],
      status: batch[3]
    }));
  };

  // Process exception data from array format to object format for table
  const processExceptionData = (exceptionsArray) => {
    if (!exceptionsArray || !Array.isArray(exceptionsArray)) return [];
    
    return exceptionsArray.map(exception => ({
      doc_id: exception[0],
      type: exception[1],
      issue: exception[2]
    }));
  };

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <Spin size="large" tip="Loading dashboard data..." />
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ textAlign: 'center', margin: '50px' }}>
        <h2>Error</h2>
        <p>{error}</p>
        <button onClick={() => window.location.reload()}>Retry</button>
      </div>
    );
  }

  const prepareDocumentTypeChartData = (typeBreakdownArray) => {
    if (!typeBreakdownArray || !Array.isArray(typeBreakdownArray)) {
      return null;
    }
    
    // Define a set of colors for the pie chart
    const backgroundColors = [
      '#1890ff', '#ff4d4f', '#52c41a', '#faad14', 
      '#722ed1', '#13c2c2', '#eb2f96', '#fa8c16', 
      '#a0d911', '#1890ff', '#f5222d', '#fa541c'
    ];
    
    // Get top 6 document types for better visualization (if more than 6 exist)
    let dataToUse = [...typeBreakdownArray];
    if (dataToUse.length > 6) {
      // Sort by count (index 1) in descending order
      dataToUse.sort((a, b) => b[1] - a[1]);
      
      // Take top 5 and combine the rest as "Other"
      const top5 = dataToUse.slice(0, 5);
      const others = dataToUse.slice(5);
      
      // Calculate total count and percentage for "Other"
      const otherCount = others.reduce((sum, item) => sum + parseInt(item[1]), 0);
      const otherPercentage = others.reduce((sum, item) => sum + parseFloat(item[2]), 0).toFixed(1);
      
      dataToUse = [...top5, ['Other', otherCount, otherPercentage]];
    }
    
    return {
      labels: dataToUse.map(item => item[0]),
      datasets: [{
        data: dataToUse.map(item => parseFloat(item[2]).toFixed(1)),
        backgroundColor: backgroundColors.slice(0, dataToUse.length)
      }]
    };
  };

  const documentTypeChartData = prepareDocumentTypeChartData(dashboardData?.document_type_breakdown);


  // Process the data for display
  const batchData = processBatchData(dashboardData?.recent_batches);
  const exceptionData = processExceptionData(dashboardData?.recent_exceptions);

  return (
    <div>
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={8}>
          <Card>
            <Statistic 
              title="Documents Today" 
              value={dashboardData?.documents_today?.[0][0] || 0} 
            />
          </Card>
        </Col>
        <Col span={8}>
          <Card>
            <Statistic 
              title="Processing Rate" 
              value={parseFloat(dashboardData?.processing_rate?.[0][0] || 0).toFixed(1)} 
              suffix="%" 
            />
          </Card>
        </Col>
        <Col span={8}>
          <Card>
            <Statistic 
              title="Pending Exceptions" 
              value={dashboardData?.pending_exceptions?.[0][0] || 0} 
              valueStyle={{ color: 'red' }} 
            />
          </Card>
        </Col>
      </Row>

      {/* <Row gutter={16}>
        <Col span={24}>
          <Card title="Document Upload">
            <Dragger {...uploadProps}>
              <p className="ant-upload-drag-icon">
                <InboxOutlined />
              </p>
              <p className="ant-upload-text">Click or drag files to this area to upload</p>
              <p className="ant-upload-hint">
                Support for single or bulk upload. Only accept document formats (.pdf, .docx, .jpg)
              </p>
            </Dragger>
          </Card>
        </Col>
      </Row> */}

      <Row gutter={16} style={{ marginTop: 16 }}>
        <Col span={24}>
          <Card title="Recent Batches">
            <Table 
              columns={batchColumns} 
              dataSource={batchData} 
              pagination={false}
              rowKey="id"
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={16} style={{ marginTop: 16 }}>
        <Col span={12}>
          <Card title="Document Type Analysis">
          {documentTypeChartData && (
              <Row>
                <Col span={12}>
                  <Pie data={documentTypeChartData} />
                </Col>
                <Col span={12}>
                  <div style={{ marginTop: '20px', maxHeight: '200px', overflowY: 'auto' }}>
                    {dashboardData?.document_type_breakdown?.map((item, index) => (
                      <div key={index} style={{ marginBottom: '5px' }}>
                        <span style={{ 
                          textTransform: 'capitalize', 
                          fontWeight: index < 5 ? 'bold' : 'normal' 
                        }}>
                          {item[0]}:
                        </span> {parseFloat(item[2]).toFixed(1)}%
                      </div>
                    ))}
                  </div>
                </Col>
              </Row>
            )}
          </Card>
        </Col>
        <Col span={12}>
          <Card title="Recent Exceptions">
            <Table 
              columns={exceptionsColumns} 
              dataSource={exceptionData} 
              pagination={false}
              rowKey="doc_id"
            />
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default Dashboard;