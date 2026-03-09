# GitHub Actions Deployment Guide

This directory contains GitHub Actions workflows for automated deployment of the Hybrid Search API.

## Files Overview

- `deploy.yml` - Main deployment workflow
- `README.md` - This documentation file

## Supported Deployment Platforms

The workflow supports multiple deployment targets:

### 1. AWS ECS (Elastic Container Service)
- **Trigger**: Push with `[deploy]` in commit message
- **Requirements**: AWS credentials and ECS cluster setup

### 2. Google Cloud Run
- **Trigger**: Push with `[deploy-gcp]` in commit message
- **Requirements**: GCP service account and project setup

### 3. Azure Container Apps
- **Trigger**: Push with `[deploy-azure]` in commit message
- **Requirements**: Azure credentials and container app setup

### 4. DigitalOcean App Platform
- **Trigger**: Push with `[deploy-do]` in commit message
- **Requirements**: DigitalOcean access token and app setup

## Setup Instructions/

### 1. GitHub Container Registry (GHCR)

The workflow automatically uses GHCR. Make sure:

1. **Package permissions**: Go to your repository Settings → Actions → General
2. **Workflow permissions**: Set to "Read and write permissions"

### 2. Required GitHub Secrets

Add these secrets to your repository (Settings → Secrets and variables → Actions):

#### Common Secrets (all platforms):
```bash
OPENAI_API_KEY=your-openai-api-key
OPENSEARCH_HOST=your-opensearch-endpoint
```

#### For AWS ECS:
```bash
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_REGION=us-east-1
AWS_ECR_URI=123456789012.dkr.ecr.us-east-1.amazonaws.com/your-repo
AWS_ECS_CLUSTER=your-cluster-name
AWS_ECS_SERVICE=your-service-name
APP_URL=https://your-app-url.com
```

#### For Google Cloud Run:
```bash
GCP_SA_KEY={"type":"service_account","project_id":"..."}  # JSON key
GCP_PROJECT_ID=your-gcp-project-id
GCP_REGION=us-central1
```

#### For Azure Container Apps:
```bash
AZURE_CREDENTIALS={"clientId":"...","clientSecret":"...","subscriptionId":"...","tenantId":"..."}
AZURE_ACR=your-acr-name
AZURE_CONTAINER_APP=your-container-app-name
AZURE_RESOURCE_GROUP=your-resource-group
OPENSEARCH_HOST=your-opensearch-host
OPENAI_API_KEY=your-openai-key
```

#### For DigitalOcean:
```bash
DIGITALOCEAN_ACCESS_TOKEN=your-do-token
DO_APP_ID=your-app-id
OPENSEARCH_HOST=your-opensearch-host
OPENAI_API_KEY=your-openai-key
AWS_ACCESS_KEY_ID=your-aws-key  # if using AWS auth
AWS_SECRET_ACCESS_KEY=your-aws-secret
```

#### Optional: Notifications
```bash
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
DISCORD_WEBHOOK=https://discord.com/api/webhooks/...
```

## How to Deploy

### Automatic Deployment

1. **Push to main/master branch** with deployment trigger:
   ```bash
   git commit -m "Add new feature [deploy]"
   git push origin main
   ```

2. **Replace `[deploy]` with your target platform**:
   - `[deploy]` - AWS ECS
   - `[deploy-gcp]` - Google Cloud Run
   - `[deploy-azure]` - Azure Container Apps
   - `[deploy-do]` - DigitalOcean App Platform

### Manual Deployment

You can also trigger deployments manually:

1. Go to your repository → Actions
2. Select "Deploy Hybrid Search API"
3. Click "Run workflow"
4. Choose the deployment target

## Workflow Jobs

### 1. Test Job
- Runs on every push and PR
- Validates Python syntax
- Tests Docker build
- **Duration**: ~2-3 minutes

### 2. Build and Push Job
- Builds optimized Docker image
- Pushes to GHCR
- Uses build cache for faster builds
- **Duration**: ~5-8 minutes

### 3. Deploy Job (Platform-specific)
- Pulls image from GHCR
- Deploys to your chosen platform
- Runs health checks
- **Duration**: ~3-10 minutes

### 4. Notify Job
- Sends deployment status to Slack/Discord
- Runs after all deployment jobs

## Environment Configuration

### Production Environment Variables

Create a `.env.prod` file in your repository root (this is gitignored):

```env
# Production Environment
FASTAPI_ENV=prod

# OpenSearch Configuration
OPENSEARCH_HOST=your-production-opensearch-domain.com
OPENSEARCH_INDEX=document-chunks
OPENSEARCH_PORT=443
OPENSEARCH_USE_SSL=true
OPENSEARCH_VERIFY_CERTS=true

# Authentication
OPENSEARCH_USERNAME=admin
OPENSEARCH_PASSWORD=your-secure-password

# OpenAI Configuration
OPENAI_API_KEY=your-production-openai-key

# API Server Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=false
API_LOG_LEVEL=info
```

## Troubleshooting

### Common Issues

1. **"Resource not accessible by integration"**
   - Check GHCR permissions in repository settings
   - Ensure workflows have write access to packages

2. **Deployment fails with authentication errors**
   - Verify all required secrets are set
   - Check that credentials have proper permissions

3. **Health check fails after deployment**
   - Ensure your OpenSearch and OpenAI services are accessible
   - Check environment variables are correctly set

4. **Build fails**
   - Verify Dockerfile syntax
   - Check that all required files are committed

### Debugging

1. **View workflow logs**:
   - Go to Actions tab in your repository
   - Click on the failed workflow run
   - Check logs for each job

2. **Test locally**:
   ```bash
   # Test Docker build
   docker build -t test .

   # Test with environment file
   docker run --env-file .env.prod -p 8000:8000 test
   ```

3. **Check deployment status**:
   - Use platform-specific CLI tools
   - Check application logs in your cloud console

## Security Best Practices

1. **Never commit secrets** to repository
2. **Use environment-specific secrets** for different deployments
3. **Rotate credentials regularly**
4. **Limit secret scope** to necessary permissions only
5. **Monitor workflow runs** for unauthorized access

## Cost Optimization

1. **Use GitHub's free minutes** for public repositories
2. **Cache Docker layers** to reduce build time
3. **Deploy only when necessary** (use commit message triggers)
4. **Choose appropriate instance sizes** for your workload
5. **Monitor usage** in GitHub Actions settings

## Advanced Configuration

### Custom Deployment Triggers

Modify the workflow to trigger on different conditions:

```yaml
on:
  push:
    branches: [ main, production ]
  schedule:
    - cron: '0 2 * * 1'  # Weekly deployment
  workflow_dispatch:  # Manual trigger
```

### Multiple Environments

Add staging environment:

```yaml
jobs:
  deploy-staging:
    if: github.ref == 'refs/heads/develop'
    # Staging deployment logic

  deploy-production:
    if: github.ref == 'refs/heads/main'
    # Production deployment logic
```

### Custom Build Steps

Add additional build steps:

```yaml
- name: Run tests
  run: |
    pip install pytest
    pytest

- name: Security scan
  uses: aquasecurity/trivy-action@master
  with:
    scan-type: 'image'
    scan-ref: 'docker.io/my/image:tag'
```

## Support

For issues with GitHub Actions:

1. Check [GitHub Actions documentation](https://docs.github.com/en/actions)
2. Review [workflow syntax](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)
3. Check [marketplace actions](https://github.com/marketplace?type=actions)

For platform-specific deployment issues, refer to:

- [AWS ECS documentation](https://docs.aws.amazon.com/ecs/)
- [Google Cloud Run docs](https://cloud.google.com/run/docs)
- [Azure Container Apps](https://docs.microsoft.com/en-us/azure/container-apps/)
- [DigitalOcean App Platform](https://docs.digitalocean.com/products/app-platform/)
