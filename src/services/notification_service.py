"""Notification Service for TruCareer System.

This service handles all user notifications including career recommendations,
test completion alerts, system updates, and personalized career insights.
Supports multiple delivery channels and notification preferences.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Set
from enum import Enum
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

from src.core.config import get_settings
from src.utils.logger import get_logger
from src.utils.exceptions import TruScholarError
from src.schemas.base import BaseModel

settings = get_settings()
logger = get_logger(__name__)


class NotificationType(Enum):
    """Types of notifications supported by the system."""
    CAREER_RECOMMENDATION = "career_recommendation"
    TEST_COMPLETION = "test_completion"
    ASSESSMENT_REMINDER = "assessment_reminder"
    CAREER_INSIGHT = "career_insight"
    MARKET_UPDATE = "market_update"
    SYSTEM_ANNOUNCEMENT = "system_announcement"
    PROGRESS_UPDATE = "progress_update"
    RECOMMENDATION_REFRESH = "recommendation_refresh"
    CAREER_OPPORTUNITY = "career_opportunity"
    SKILL_RECOMMENDATION = "skill_recommendation"


class NotificationChannel(Enum):
    """Available notification delivery channels."""
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    IN_APP = "in_app"
    WHATSAPP = "whatsapp"


class NotificationPriority(Enum):
    """Notification priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class NotificationStatus(Enum):
    """Notification delivery status."""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    CLICKED = "clicked"
    DISMISSED = "dismissed"


@dataclass
class NotificationTemplate:
    """Template for notification messages."""
    subject: str
    body: str
    action_url: Optional[str] = None
    action_text: Optional[str] = None
    variables: Optional[Dict[str, Any]] = None


@dataclass
class NotificationRecipient:
    """Notification recipient information."""
    user_id: str
    email: Optional[str] = None
    phone: Optional[str] = None
    push_token: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None
    timezone: str = "Asia/Kolkata"
    language: str = "en"


@dataclass
class NotificationPayload:
    """Complete notification payload."""
    id: str
    type: NotificationType
    recipient: NotificationRecipient
    channels: List[NotificationChannel]
    priority: NotificationPriority
    template: NotificationTemplate
    scheduled_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


class NotificationError(TruScholarError):
    """Exception raised when notification operations fail."""
    pass


class NotificationChannel_Handler(ABC):
    """Abstract base class for notification channel handlers."""
    
    @abstractmethod
    async def send_notification(self, payload: NotificationPayload) -> Dict[str, Any]:
        """Send notification through this channel."""
        pass
    
    @abstractmethod
    async def verify_delivery(self, notification_id: str) -> Dict[str, Any]:
        """Verify notification delivery status."""
        pass


class EmailNotificationHandler(NotificationChannel_Handler):
    """Email notification handler."""
    
    def __init__(self, smtp_config: Dict[str, Any]):
        self.smtp_config = smtp_config
        self.templates = self._load_email_templates()
    
    async def send_notification(self, payload: NotificationPayload) -> Dict[str, Any]:
        """Send email notification."""
        try:
            logger.info(f"Sending email notification {payload.id} to {payload.recipient.email}")
            
            # Prepare email content
            subject = self._render_template(payload.template.subject, payload.template.variables)
            body = self._render_template(payload.template.body, payload.template.variables)
            
            # Add personalization
            body = self._add_personalization(body, payload.recipient)
            
            # Send email (implementation would use actual SMTP)
            email_result = await self._send_email(
                to_email=payload.recipient.email,
                subject=subject,
                body=body,
                template_type=payload.type.value
            )
            
            return {
                "status": "sent",
                "channel": NotificationChannel.EMAIL.value,
                "message_id": email_result.get("message_id"),
                "sent_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {str(e)}")
            return {
                "status": "failed",
                "channel": NotificationChannel.EMAIL.value,
                "error": str(e),
                "failed_at": datetime.utcnow().isoformat()
            }
    
    async def verify_delivery(self, notification_id: str) -> Dict[str, Any]:
        """Verify email delivery status."""
        # Implementation would check with email service provider
        return {
            "notification_id": notification_id,
            "status": "delivered",
            "delivered_at": datetime.utcnow().isoformat()
        }
    
    def _render_template(self, template: str, variables: Optional[Dict[str, Any]]) -> str:
        """Render template with variables."""
        if not variables:
            return template
        
        rendered = template
        for key, value in variables.items():
            rendered = rendered.replace(f"{{{key}}}", str(value))
        
        return rendered
    
    def _add_personalization(self, body: str, recipient: NotificationRecipient) -> str:
        """Add personalization to email body."""
        # Add greeting based on time of day
        current_hour = datetime.now().hour
        if 5 <= current_hour < 12:
            greeting = "Good morning"
        elif 12 <= current_hour < 17:
            greeting = "Good afternoon"
        else:
            greeting = "Good evening"
        
        # Add personalized greeting
        if not body.startswith(greeting):
            body = f"{greeting},\n\n{body}"
        
        return body
    
    async def _send_email(self, to_email: str, subject: str, body: str, template_type: str) -> Dict[str, Any]:
        """Send email using SMTP or email service."""
        # Mock implementation - would use actual email service
        return {
            "message_id": f"email_{datetime.utcnow().timestamp()}",
            "status": "sent"
        }
    
    def _load_email_templates(self) -> Dict[str, NotificationTemplate]:
        """Load email templates."""
        return {
            "career_recommendation": NotificationTemplate(
                subject="Your Personalized Career Recommendations are Ready! 🚀",
                body="""Hi there!

Great news! We've analyzed your RAISEC assessment results and prepared personalized career recommendations just for you.

Here's what we found:
• {recommendation_count} career matches based on your profile
• {top_match} is your strongest match with {match_score}% compatibility
• Your dominant personality dimensions: {top_dimensions}

Ready to explore your career possibilities?

{action_text}

Best regards,
TruCareer Team""",
                action_url="/dashboard/recommendations",
                action_text="View My Recommendations"
            ),
            
            "test_completion": NotificationTemplate(
                subject="RAISEC Assessment Completed Successfully! ✅",
                body="""Congratulations!

You've successfully completed your RAISEC career assessment. Your results are now being processed to generate personalized career recommendations.

Assessment Summary:
• Completed: {completion_date}
• Questions answered: {questions_answered}
• Assessment type: {assessment_type}

Your personalized career recommendations will be ready in a few minutes. We'll notify you as soon as they're available.

{action_text}

Thank you for choosing TruCareer!""",
                action_url="/dashboard",
                action_text="View Dashboard"
            ),
            
            "career_insight": NotificationTemplate(
                subject="New Career Insights Available for You! 💡",
                body="""Hello!

We have new personalized career insights based on your profile and current market trends:

Key Insights:
• {insight_title}
• Market trend: {market_trend}
• Recommendation: {recommendation}

These insights can help you make informed decisions about your career path.

{action_text}

Stay ahead with TruCareer!""",
                action_url="/insights",
                action_text="Explore Insights"
            )
        }


class SMSNotificationHandler(NotificationChannel_Handler):
    """SMS notification handler."""
    
    def __init__(self, sms_config: Dict[str, Any]):
        self.sms_config = sms_config
    
    async def send_notification(self, payload: NotificationPayload) -> Dict[str, Any]:
        """Send SMS notification."""
        try:
            logger.info(f"Sending SMS notification {payload.id} to {payload.recipient.phone}")
            
            # Prepare SMS content (keep it short)
            message = self._prepare_sms_content(payload)
            
            # Send SMS (implementation would use actual SMS service)
            sms_result = await self._send_sms(
                to_phone=payload.recipient.phone,
                message=message
            )
            
            return {
                "status": "sent",
                "channel": NotificationChannel.SMS.value,
                "message_id": sms_result.get("message_id"),
                "sent_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to send SMS notification: {str(e)}")
            return {
                "status": "failed",
                "channel": NotificationChannel.SMS.value,
                "error": str(e),
                "failed_at": datetime.utcnow().isoformat()
            }
    
    async def verify_delivery(self, notification_id: str) -> Dict[str, Any]:
        """Verify SMS delivery status."""
        # Implementation would check with SMS service provider
        return {
            "notification_id": notification_id,
            "status": "delivered",
            "delivered_at": datetime.utcnow().isoformat()
        }
    
    def _prepare_sms_content(self, payload: NotificationPayload) -> str:
        """Prepare SMS content (short format)."""
        templates = {
            NotificationType.CAREER_RECOMMENDATION: "TruCareer: Your career recommendations are ready! {recommendation_count} matches found. View: {action_url}",
            NotificationType.TEST_COMPLETION: "TruCareer: Assessment completed successfully! Results processing. Check dashboard: {action_url}",
            NotificationType.CAREER_INSIGHT: "TruCareer: New career insights available. {insight_title}. View: {action_url}"
        }
        
        template = templates.get(payload.type, "TruCareer: You have a new notification. Check your dashboard.")
        
        # Replace variables
        if payload.template.variables:
            for key, value in payload.template.variables.items():
                template = template.replace(f"{{{key}}}", str(value))
        
        return template
    
    async def _send_sms(self, to_phone: str, message: str) -> Dict[str, Any]:
        """Send SMS using SMS service."""
        # Mock implementation - would use actual SMS service
        return {
            "message_id": f"sms_{datetime.utcnow().timestamp()}",
            "status": "sent"
        }


class PushNotificationHandler(NotificationChannel_Handler):
    """Push notification handler."""
    
    def __init__(self, push_config: Dict[str, Any]):
        self.push_config = push_config
    
    async def send_notification(self, payload: NotificationPayload) -> Dict[str, Any]:
        """Send push notification."""
        try:
            logger.info(f"Sending push notification {payload.id} to token {payload.recipient.push_token}")
            
            # Prepare push notification content
            push_data = self._prepare_push_content(payload)
            
            # Send push notification
            push_result = await self._send_push(
                push_token=payload.recipient.push_token,
                title=push_data["title"],
                body=push_data["body"],
                data=push_data.get("data", {})
            )
            
            return {
                "status": "sent",
                "channel": NotificationChannel.PUSH.value,
                "message_id": push_result.get("message_id"),
                "sent_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to send push notification: {str(e)}")
            return {
                "status": "failed",
                "channel": NotificationChannel.PUSH.value,
                "error": str(e),
                "failed_at": datetime.utcnow().isoformat()
            }
    
    async def verify_delivery(self, notification_id: str) -> Dict[str, Any]:
        """Verify push notification delivery."""
        # Implementation would check with push service provider
        return {
            "notification_id": notification_id,
            "status": "delivered",
            "delivered_at": datetime.utcnow().isoformat()
        }
    
    def _prepare_push_content(self, payload: NotificationPayload) -> Dict[str, Any]:
        """Prepare push notification content."""
        templates = {
            NotificationType.CAREER_RECOMMENDATION: {
                "title": "Career Recommendations Ready! 🚀",
                "body": "{recommendation_count} personalized matches found",
                "data": {"type": "career_recommendations", "action_url": payload.template.action_url}
            },
            NotificationType.TEST_COMPLETION: {
                "title": "Assessment Completed! ✅",
                "body": "Your RAISEC results are being processed",
                "data": {"type": "test_completion", "action_url": payload.template.action_url}
            },
            NotificationType.CAREER_INSIGHT: {
                "title": "New Career Insights 💡",
                "body": "Personalized insights available",
                "data": {"type": "career_insights", "action_url": payload.template.action_url}
            }
        }
        
        template = templates.get(payload.type, {
            "title": "TruCareer Notification",
            "body": "You have a new notification",
            "data": {"type": "general"}
        })
        
        # Replace variables in body
        if payload.template.variables:
            for key, value in payload.template.variables.items():
                template["body"] = template["body"].replace(f"{{{key}}}", str(value))
        
        return template
    
    async def _send_push(self, push_token: str, title: str, body: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send push notification using FCM or similar service."""
        # Mock implementation - would use actual push service
        return {
            "message_id": f"push_{datetime.utcnow().timestamp()}",
            "status": "sent"
        }


class InAppNotificationHandler(NotificationChannel_Handler):
    """In-app notification handler."""
    
    def __init__(self, database_client):
        self.db = database_client
    
    async def send_notification(self, payload: NotificationPayload) -> Dict[str, Any]:
        """Store in-app notification in database."""
        try:
            logger.info(f"Creating in-app notification {payload.id} for user {payload.recipient.user_id}")
            
            # Prepare in-app notification data
            notification_data = {
                "id": payload.id,
                "user_id": payload.recipient.user_id,
                "type": payload.type.value,
                "title": payload.template.subject,
                "message": payload.template.body,
                "action_url": payload.template.action_url,
                "action_text": payload.template.action_text,
                "priority": payload.priority.value,
                "status": NotificationStatus.SENT.value,
                "created_at": datetime.utcnow(),
                "read_at": None,
                "metadata": payload.metadata or {}
            }
            
            # Replace variables in message
            if payload.template.variables:
                for key, value in payload.template.variables.items():
                    notification_data["message"] = notification_data["message"].replace(f"{{{key}}}", str(value))
            
            # Store in database
            result = await self._store_notification(notification_data)
            
            return {
                "status": "sent",
                "channel": NotificationChannel.IN_APP.value,
                "notification_id": payload.id,
                "stored_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to create in-app notification: {str(e)}")
            return {
                "status": "failed",
                "channel": NotificationChannel.IN_APP.value,
                "error": str(e),
                "failed_at": datetime.utcnow().isoformat()
            }
    
    async def verify_delivery(self, notification_id: str) -> Dict[str, Any]:
        """Verify in-app notification is stored."""
        # Check if notification exists in database
        notification = await self._get_notification(notification_id)
        if notification:
            return {
                "notification_id": notification_id,
                "status": "delivered",
                "delivered_at": notification.get("created_at", datetime.utcnow()).isoformat()
            }
        else:
            return {
                "notification_id": notification_id,
                "status": "failed",
                "error": "Notification not found in database"
            }
    
    async def _store_notification(self, notification_data: Dict[str, Any]) -> Dict[str, Any]:
        """Store notification in database."""
        # Mock implementation - would use actual database
        return {"id": notification_data["id"], "stored": True}
    
    async def _get_notification(self, notification_id: str) -> Optional[Dict[str, Any]]:
        """Get notification from database."""
        # Mock implementation - would query actual database
        return {"id": notification_id, "created_at": datetime.utcnow()}


class NotificationService:
    """Main notification service orchestrator."""
    
    def __init__(self, 
                 email_config: Optional[Dict[str, Any]] = None,
                 sms_config: Optional[Dict[str, Any]] = None,
                 push_config: Optional[Dict[str, Any]] = None,
                 database_client=None):
        """Initialize notification service with channel handlers."""
        
        self.handlers = {}
        
        # Initialize channel handlers
        if email_config:
            self.handlers[NotificationChannel.EMAIL] = EmailNotificationHandler(email_config)
        
        if sms_config:
            self.handlers[NotificationChannel.SMS] = SMSNotificationHandler(sms_config)
        
        if push_config:
            self.handlers[NotificationChannel.PUSH] = PushNotificationHandler(push_config)
        
        if database_client:
            self.handlers[NotificationChannel.IN_APP] = InAppNotificationHandler(database_client)
        
        # Notification queue for batch processing
        self.notification_queue = []
        self.processing_enabled = True
        
        # Statistics tracking
        self.stats = {
            "total_sent": 0,
            "total_failed": 0,
            "by_channel": {channel.value: {"sent": 0, "failed": 0} for channel in NotificationChannel},
            "by_type": {ntype.value: {"sent": 0, "failed": 0} for ntype in NotificationType}
        }
    
    async def send_notification(self, 
                              notification_type: NotificationType,
                              recipient: NotificationRecipient,
                              template_variables: Optional[Dict[str, Any]] = None,
                              channels: Optional[List[NotificationChannel]] = None,
                              priority: NotificationPriority = NotificationPriority.MEDIUM,
                              scheduled_at: Optional[datetime] = None) -> Dict[str, Any]:
        """Send notification through specified channels."""
        
        # Generate notification ID
        notification_id = f"notif_{int(datetime.utcnow().timestamp())}_{recipient.user_id}"
        
        # Determine channels based on user preferences
        if not channels:
            channels = self._determine_channels(recipient, notification_type, priority)
        
        # Get template for notification type
        template = self._get_template(notification_type, template_variables)
        
        # Create notification payload
        payload = NotificationPayload(
            id=notification_id,
            type=notification_type,
            recipient=recipient,
            channels=channels,
            priority=priority,
            template=template,
            scheduled_at=scheduled_at,
            metadata={"created_at": datetime.utcnow().isoformat()}
        )
        
        # Send notification
        if scheduled_at and scheduled_at > datetime.utcnow():
            # Schedule for later delivery
            return await self._schedule_notification(payload)
        else:
            # Send immediately
            return await self._send_notification_now(payload)
    
    async def send_career_recommendation_notification(self,
                                                    user_id: str,
                                                    email: str,
                                                    recommendation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send career recommendation notification."""
        
        recipient = NotificationRecipient(
            user_id=user_id,
            email=email
        )
        
        template_variables = {
            "recommendation_count": recommendation_data.get("count", 0),
            "top_match": recommendation_data.get("top_match", ""),
            "match_score": recommendation_data.get("match_score", 0),
            "top_dimensions": recommendation_data.get("top_dimensions", "")
        }
        
        return await self.send_notification(
            notification_type=NotificationType.CAREER_RECOMMENDATION,
            recipient=recipient,
            template_variables=template_variables,
            priority=NotificationPriority.HIGH
        )
    
    async def send_test_completion_notification(self,
                                              user_id: str,
                                              email: str,
                                              test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send test completion notification."""
        
        recipient = NotificationRecipient(
            user_id=user_id,
            email=email
        )
        
        template_variables = {
            "completion_date": test_data.get("completion_date", datetime.utcnow().strftime("%B %d, %Y")),
            "questions_answered": test_data.get("questions_answered", 0),
            "assessment_type": test_data.get("assessment_type", "RAISEC Career Assessment")
        }
        
        return await self.send_notification(
            notification_type=NotificationType.TEST_COMPLETION,
            recipient=recipient,
            template_variables=template_variables,
            priority=NotificationPriority.HIGH
        )
    
    async def send_career_insight_notification(self,
                                             user_id: str,
                                             email: str,
                                             insight_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send career insight notification."""
        
        recipient = NotificationRecipient(
            user_id=user_id,
            email=email
        )
        
        template_variables = {
            "insight_title": insight_data.get("title", "New Career Insight"),
            "market_trend": insight_data.get("market_trend", ""),
            "recommendation": insight_data.get("recommendation", "")
        }
        
        return await self.send_notification(
            notification_type=NotificationType.CAREER_INSIGHT,
            recipient=recipient,
            template_variables=template_variables,
            priority=NotificationPriority.MEDIUM
        )
    
    async def send_bulk_notifications(self, notifications: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Send multiple notifications in batch."""
        
        results = []
        success_count = 0
        failure_count = 0
        
        # Process notifications in batches
        batch_size = 10
        for i in range(0, len(notifications), batch_size):
            batch = notifications[i:i + batch_size]
            
            # Send batch concurrently
            batch_tasks = []
            for notif_data in batch:
                task = self.send_notification(**notif_data)
                batch_tasks.append(task)
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Process batch results
            for result in batch_results:
                if isinstance(result, Exception):
                    failure_count += 1
                    results.append({"status": "failed", "error": str(result)})
                else:
                    success_count += 1
                    results.append(result)
        
        return {
            "total_processed": len(notifications),
            "success_count": success_count,
            "failure_count": failure_count,
            "results": results
        }
    
    async def get_user_notifications(self, 
                                   user_id: str,
                                   limit: int = 20,
                                   offset: int = 0,
                                   unread_only: bool = False) -> Dict[str, Any]:
        """Get user's in-app notifications."""
        
        in_app_handler = self.handlers.get(NotificationChannel.IN_APP)
        if not in_app_handler:
            return {"notifications": [], "total": 0}
        
        # Mock implementation - would query actual database
        notifications = [
            {
                "id": f"notif_{i}",
                "type": "career_recommendation",
                "title": f"Notification {i}",
                "message": f"Message content {i}",
                "created_at": (datetime.utcnow() - timedelta(hours=i)).isoformat(),
                "read_at": None if i < 3 else (datetime.utcnow() - timedelta(hours=i-1)).isoformat()
            }
            for i in range(1, 11)
        ]
        
        if unread_only:
            notifications = [n for n in notifications if n["read_at"] is None]
        
        # Apply pagination
        paginated_notifications = notifications[offset:offset + limit]
        
        return {
            "notifications": paginated_notifications,
            "total": len(notifications),
            "unread_count": len([n for n in notifications if n["read_at"] is None])
        }
    
    async def mark_notification_as_read(self, notification_id: str, user_id: str) -> Dict[str, Any]:
        """Mark notification as read."""
        
        # Mock implementation - would update actual database
        return {
            "notification_id": notification_id,
            "status": "marked_as_read",
            "read_at": datetime.utcnow().isoformat()
        }
    
    async def get_notification_statistics(self) -> Dict[str, Any]:
        """Get notification service statistics."""
        
        return {
            "total_notifications_sent": self.stats["total_sent"],
            "total_notifications_failed": self.stats["total_failed"],
            "success_rate": self.stats["total_sent"] / max(1, self.stats["total_sent"] + self.stats["total_failed"]),
            "by_channel": self.stats["by_channel"],
            "by_type": self.stats["by_type"],
            "active_channels": list(self.handlers.keys()),
            "queue_size": len(self.notification_queue)
        }
    
    def _determine_channels(self, 
                          recipient: NotificationRecipient,
                          notification_type: NotificationType,
                          priority: NotificationPriority) -> List[NotificationChannel]:
        """Determine notification channels based on user preferences and priority."""
        
        # Default channel preferences
        default_channels = {
            NotificationType.CAREER_RECOMMENDATION: [NotificationChannel.EMAIL, NotificationChannel.IN_APP],
            NotificationType.TEST_COMPLETION: [NotificationChannel.EMAIL, NotificationChannel.PUSH, NotificationChannel.IN_APP],
            NotificationType.CAREER_INSIGHT: [NotificationChannel.EMAIL, NotificationChannel.IN_APP],
            NotificationType.SYSTEM_ANNOUNCEMENT: [NotificationChannel.IN_APP],
            NotificationType.ASSESSMENT_REMINDER: [NotificationChannel.EMAIL, NotificationChannel.SMS]
        }
        
        channels = default_channels.get(notification_type, [NotificationChannel.IN_APP])
        
        # Adjust based on priority
        if priority == NotificationPriority.URGENT:
            if NotificationChannel.SMS not in channels and recipient.phone:
                channels.append(NotificationChannel.SMS)
        
        # Filter based on available handlers and recipient info
        available_channels = []
        for channel in channels:
            if channel in self.handlers:
                if channel == NotificationChannel.EMAIL and recipient.email:
                    available_channels.append(channel)
                elif channel == NotificationChannel.SMS and recipient.phone:
                    available_channels.append(channel)
                elif channel == NotificationChannel.PUSH and recipient.push_token:
                    available_channels.append(channel)
                elif channel == NotificationChannel.IN_APP:
                    available_channels.append(channel)
        
        return available_channels or [NotificationChannel.IN_APP]
    
    def _get_template(self, notification_type: NotificationType, variables: Optional[Dict[str, Any]]) -> NotificationTemplate:
        """Get notification template for the given type."""
        
        # Get template from email handler (contains all templates)
        email_handler = self.handlers.get(NotificationChannel.EMAIL)
        if email_handler and hasattr(email_handler, 'templates'):
            template = email_handler.templates.get(notification_type.value)
            if template:
                template.variables = variables
                return template
        
        # Fallback template
        return NotificationTemplate(
            subject="TruCareer Notification",
            body="You have a new notification from TruCareer.",
            variables=variables
        )
    
    async def _send_notification_now(self, payload: NotificationPayload) -> Dict[str, Any]:
        """Send notification immediately through all specified channels."""
        
        results = []
        
        # Send through each channel
        for channel in payload.channels:
            handler = self.handlers.get(channel)
            if handler:
                try:
                    result = await handler.send_notification(payload)
                    results.append(result)
                    
                    # Update statistics
                    if result["status"] == "sent":
                        self.stats["total_sent"] += 1
                        self.stats["by_channel"][channel.value]["sent"] += 1
                        self.stats["by_type"][payload.type.value]["sent"] += 1
                    else:
                        self.stats["total_failed"] += 1
                        self.stats["by_channel"][channel.value]["failed"] += 1
                        self.stats["by_type"][payload.type.value]["failed"] += 1
                        
                except Exception as e:
                    logger.error(f"Failed to send notification through {channel.value}: {str(e)}")
                    results.append({
                        "status": "failed",
                        "channel": channel.value,
                        "error": str(e)
                    })
                    
                    self.stats["total_failed"] += 1
                    self.stats["by_channel"][channel.value]["failed"] += 1
                    self.stats["by_type"][payload.type.value]["failed"] += 1
            else:
                results.append({
                    "status": "failed",
                    "channel": channel.value,
                    "error": "Handler not available"
                })
        
        return {
            "notification_id": payload.id,
            "status": "processed",
            "channels_attempted": len(payload.channels),
            "results": results,
            "sent_at": datetime.utcnow().isoformat()
        }
    
    async def _schedule_notification(self, payload: NotificationPayload) -> Dict[str, Any]:
        """Schedule notification for later delivery."""
        
        # Add to queue with scheduled time
        self.notification_queue.append(payload)
        
        logger.info(f"Notification {payload.id} scheduled for {payload.scheduled_at}")
        
        return {
            "notification_id": payload.id,
            "status": "scheduled",
            "scheduled_at": payload.scheduled_at.isoformat(),
            "channels": [c.value for c in payload.channels]
        }
    
    async def process_scheduled_notifications(self):
        """Process notifications that are scheduled for delivery."""
        
        if not self.processing_enabled:
            return
        
        current_time = datetime.utcnow()
        due_notifications = []
        
        # Find notifications due for delivery
        for notification in self.notification_queue:
            if notification.scheduled_at and notification.scheduled_at <= current_time:
                due_notifications.append(notification)
        
        # Remove from queue and send
        for notification in due_notifications:
            self.notification_queue.remove(notification)
            await self._send_notification_now(notification)
        
        logger.info(f"Processed {len(due_notifications)} scheduled notifications")


# Factory function for creating notification service
def create_notification_service(config: Optional[Dict[str, Any]] = None) -> NotificationService:
    """Create and configure notification service."""
    
    if not config:
        config = {
            "email": {
                "smtp_host": settings.SMTP_HOST,
                "smtp_port": settings.SMTP_PORT,
                "smtp_user": settings.SMTP_USER,
                "smtp_password": settings.SMTP_PASSWORD
            },
            "sms": {
                "api_key": settings.SMS_API_KEY,
                "sender_id": settings.SMS_SENDER_ID
            },
            "push": {
                "fcm_server_key": settings.FCM_SERVER_KEY
            }
        }
    
    return NotificationService(
        email_config=config.get("email"),
        sms_config=config.get("sms"),
        push_config=config.get("push"),
        database_client=None  # Would inject actual database client
    )