import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  Table, Card, Button, Modal, Form, Select, DatePicker, Input, Tag 
} from 'antd';
import { FilterOutlined, FileSearchOutlined, DownloadOutlined } from '@ant-design/icons';

const { Option } = Select;
const { RangePicker } = DatePicker;

const ProcessedDocuments = () => {
  const [isFilterModalVisible, setIsFilterModalVisible] = useState(false);
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);

  const fetchDocuments = async () => {
    try {
      const response = await axios.get('http://localhost:3000/api/processed-documents');
      const rawData = response.data.processed_documents;

    const transformedData = rawData.map((doc, index) => ({
      key: doc[0], // id
      documentId: doc[1],
      fileName: doc[2],
      type: doc[3],
      processedDate: doc[4],
      batchId: doc[5],
      status: doc[6],
    }));

    setDocuments(transformedData);
    } catch (error) {
      console.error('Error fetching documents:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDocuments();
  }, []);

  const columns = [
    {
      title: 'Document ID',
      dataIndex: 'documentId',
      key: 'documentId',
    },
    {
      title: 'File Name',
      dataIndex: 'fileName',
      key: 'fileName',
    },
    {
      title: 'Type',
      dataIndex: 'type',
      key: 'type',
      filters: [
        { text: 'Claim', value: 'Claim' },
        { text: 'Agency', value: 'Agency' },
        { text: 'Mortgage', value: 'Mortgage' },
      ],
      onFilter: (value, record) => record.type === value,
    },
    {
      title: 'Processed Date',
      dataIndex: 'processedDate',
      key: 'processedDate',
    },
    {
      title: 'Batch ID',
      dataIndex: 'batchId',
      key: 'batchId',
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status) => {
        const color = {
          'Processed': 'green',
          'Verified': 'blue'
        }[status] || 'default';
        return <Tag color={color}>{status}</Tag>;
      }
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_, record) => (
        <>
          <Button 
            size="small" 
            icon={<FileSearchOutlined />}
            onClick={() => handleViewDocument(record)}
            style={{ marginRight: 8 }}
          >
            View
          </Button>
          <Button 
            size="small" 
            icon={<DownloadOutlined />}
            type="primary"
          >
            Download
          </Button>
        </>
      )
    }
  ];

  const handleViewDocument = (document) => {
    Modal.info({
      title: `Document Details: ${document.documentId}`,
      content: (
        <div>
          <p>File Name: {document.fileName}</p>
          <p>Type: {document.type}</p>
          <p>Processed Date: {document.processedDate}</p>
          <p>Batch ID: {document.batchId}</p>
          <p>Status: {document.status}</p>
        </div>
      ),
      onOk() {},
    });
  };

  const handleFilterSubmit = (values) => {
    console.log('Filter values:', values);
    setIsFilterModalVisible(false);
    // You can call fetchDocuments() with query params if backend supports
  };

  return (
    <Card 
      title="Processed Documents" 
      extra={
        <Button 
          icon={<FilterOutlined />} 
          onClick={() => setIsFilterModalVisible(true)}
        >
          Filters
        </Button>
      }
    >
      <Table 
        columns={columns} 
        dataSource={documents}
        loading={loading}
        pagination={{ pageSize: 5 }}
      />

      <Modal
        title="Filter Processed Documents"
        visible={isFilterModalVisible}
        onCancel={() => setIsFilterModalVisible(false)}
        footer={null}
      >
        <Form onFinish={handleFilterSubmit}>
          <Form.Item name="documentType" label="Document Type">
            <Select placeholder="Select document type">
              <Option value="Claim">Claim</Option>
              <Option value="Agency">Agency</Option>
              <Option value="Mortgage">Mortgage</Option>
            </Select>
          </Form.Item>
          <Form.Item name="processedDateRange" label="Processed Date Range">
            <RangePicker />
          </Form.Item>
          <Form.Item name="documentId" label="Document ID">
            <Input placeholder="Enter Document ID" />
          </Form.Item>
          <Form.Item>
            <Button type="primary" htmlType="submit">
              Apply Filters
            </Button>
          </Form.Item>
        </Form>
      </Modal>
    </Card>
  );
};

export default ProcessedDocuments;
