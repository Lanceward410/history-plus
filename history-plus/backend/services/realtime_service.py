"""
Real-time event system for live dashboard updates
Provides WebSocket and Server-Sent Events for real-time data streaming
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import logging

logger = logging.getLogger(__name__)

try:
    from flask_socketio import SocketIO, emit, join_room, leave_room
    from flask import request
    SOCKETIO_AVAILABLE = True
except ImportError:
    logger.warning("Flask-SocketIO not available. Real-time features will be limited.")
    SOCKETIO_AVAILABLE = False

from services.database_service import DatabaseEvent, DatabaseChangeType

class EventType(Enum):
    """Types of real-time events"""
    DATABASE_CHANGE = "database_change"
    TASK_UPDATE = "task_update"
    CATEGORY_LEARNED = "category_learned"
    SEMANTIC_SEARCH_RESULT = "semantic_search_result"
    ANALYTICS_UPDATE = "analytics_update"
    PERFORMANCE_ALERT = "performance_alert"
    USER_ACTIVITY = "user_activity"

@dataclass
class RealtimeEvent:
    """Real-time event structure"""
    event_type: EventType
    data: Dict[str, Any]
    timestamp: datetime = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'event_type': self.event_type.value,
            'data': self.data,
            'timestamp': self.timestamp.isoformat(),
            'session_id': self.session_id,
            'user_id': self.user_id
        }

class RealtimeService:
    """
    Real-time service for live dashboard updates
    Handles WebSocket connections and event broadcasting
    """
    
    def __init__(self, socketio: Optional[SocketIO] = None):
        self.socketio = socketio
        self.connected_clients: Dict[str, Dict[str, Any]] = {}
        self.room_subscriptions: Dict[str, Set[str]] = {}
        self.event_history: List[RealtimeEvent] = []
        self.max_history_size = 1000
        
        # Performance monitoring
        self.event_stats = {
            'events_sent': 0,
            'events_queued': 0,
            'connections_active': 0,
            'last_activity': datetime.now()
        }
        
        # Fallback for when SocketIO is not available
        self.event_callbacks: Dict[str, List[callable]] = {}
        
        # Setup WebSocket event handlers if available
        if self.socketio and SOCKETIO_AVAILABLE:
            self._setup_websocket_handlers()
    
    def _setup_websocket_handlers(self):
        """Setup WebSocket event handlers"""
        if not self.socketio:
            return
            
        try:
            @self.socketio.on('connect')
            def handle_connect():
                try:
                    client_id = request.sid
                    self.connected_clients[client_id] = {
                        'connected_at': datetime.now(),
                        'subscriptions': set(),
                        'last_activity': datetime.now()
                    }
                    self.event_stats['connections_active'] = len(self.connected_clients)
                    
                    # Send connection confirmation
                    emit('connection_confirmed', {
                        'client_id': client_id,
                        'server_time': datetime.now().isoformat(),
                        'available_rooms': list(self.room_subscriptions.keys())
                    })
                    logger.info(f"Client {client_id} connected")
                except Exception as e:
                    logger.error(f"Error handling WebSocket connect: {e}")
            
            @self.socketio.on('disconnect')
            def handle_disconnect():
                try:
                    client_id = request.sid
                    if client_id in self.connected_clients:
                        # Remove from all rooms
                        client_subscriptions = self.connected_clients[client_id]['subscriptions']
                        for room in client_subscriptions:
                            leave_room(room)
                            if room in self.room_subscriptions:
                                self.room_subscriptions[room].discard(client_id)
                        
                        del self.connected_clients[client_id]
                        self.event_stats['connections_active'] = len(self.connected_clients)
                        logger.info(f"Client {client_id} disconnected")
                except Exception as e:
                    logger.error(f"Error handling WebSocket disconnect: {e}")
            
            @self.socketio.on('subscribe')
            def handle_subscribe(data):
                try:
                    client_id = request.sid
                    room = data.get('room')
                    
                    if room and client_id in self.connected_clients:
                        join_room(room)
                        self.connected_clients[client_id]['subscriptions'].add(room)
                        
                        if room not in self.room_subscriptions:
                            self.room_subscriptions[room] = set()
                        self.room_subscriptions[room].add(client_id)
                        
                        emit('subscribed', {'room': room, 'status': 'success'})
                        logger.info(f"Client {client_id} subscribed to room {room}")
                        
                        # Send recent events for this room
                        recent_events = self._get_recent_events_for_room(room)
                        if recent_events:
                            emit('recent_events', {'room': room, 'events': recent_events})
                except Exception as e:
                    logger.error(f"Error handling subscribe: {e}")
                    emit('subscribed', {'room': data.get('room'), 'status': 'error', 'message': str(e)})
            
            @self.socketio.on('unsubscribe')
            def handle_unsubscribe(data):
                try:
                    client_id = request.sid
                    room = data.get('room')
                    
                    if room and client_id in self.connected_clients:
                        leave_room(room)
                        self.connected_clients[client_id]['subscriptions'].discard(room)
                        
                        if room in self.room_subscriptions:
                            self.room_subscriptions[room].discard(client_id)
                        
                        emit('unsubscribed', {'room': room, 'status': 'success'})
                        logger.info(f"Client {client_id} unsubscribed from room {room}")
                except Exception as e:
                    logger.error(f"Error handling unsubscribe: {e}")
                    emit('unsubscribed', {'room': data.get('room'), 'status': 'error', 'message': str(e)})
            
            @self.socketio.on('ping')
            def handle_ping():
                try:
                    client_id = request.sid
                    if client_id in self.connected_clients:
                        self.connected_clients[client_id]['last_activity'] = datetime.now()
                    emit('pong', {'timestamp': datetime.now().isoformat()})
                except Exception as e:
                    logger.error(f"Error handling ping: {e}")
            
            @self.socketio.on('request_dashboard_data')
            def handle_dashboard_data_request(data):
                """Handle dashboard data requests"""
                try:
                    client_id = request.sid
                    data_type = data.get('type', 'overview')
                    
                    # Generate dashboard data based on type
                    if data_type == 'overview':
                        dashboard_data = self._generate_overview_data()
                    elif data_type == 'analytics':
                        dashboard_data = self._generate_analytics_data()
                    elif data_type == 'categories':
                        dashboard_data = self._generate_category_data()
                    else:
                        dashboard_data = {'error': f'Unknown data type: {data_type}'}
                    
                    emit('dashboard_data', {
                        'type': data_type,
                        'data': dashboard_data,
                        'timestamp': datetime.now().isoformat()
                    })
                except Exception as e:
                    logger.error(f"Error handling dashboard data request: {e}")
                    emit('dashboard_data', {
                        'type': data.get('type', 'unknown'),
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    })
            
            logger.info("WebSocket handlers setup complete")
            
        except Exception as e:
            logger.error(f"Error setting up WebSocket handlers: {e}")
    
    def broadcast_event(self, event: RealtimeEvent, room: str = None):
        """
        Broadcast event to connected clients
        
        Args:
            event: RealtimeEvent to broadcast
            room: Optional room to broadcast to (broadcasts to all if None)
        """
        try:
            self.event_stats['events_sent'] += 1
            self.event_stats['last_activity'] = datetime.now()
            
            # Add to event history
            self.event_history.append(event)
            if len(self.event_history) > self.max_history_size:
                self.event_history = self.event_history[-self.max_history_size:]
            
            # Broadcast event via WebSocket if available
            if self.socketio and SOCKETIO_AVAILABLE:
                event_data = event.to_dict()
                
                if room:
                    self.socketio.emit('realtime_event', event_data, room=room)
                else:
                    self.socketio.emit('realtime_event', event_data)
            
            # Also trigger any registered callbacks (fallback mechanism)
            event_type = event.event_type.value
            for callback in self.event_callbacks.get(event_type, []):
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Error in event callback: {e}")
            
            for callback in self.event_callbacks.get('*', []):
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Error in global event callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error broadcasting event: {e}")
    
    def add_event_callback(self, event_type: str, callback: callable):
        """Add callback for events (fallback when WebSocket not available)"""
        if event_type not in self.event_callbacks:
            self.event_callbacks[event_type] = []
        self.event_callbacks[event_type].append(callback)
    
    def remove_event_callback(self, event_type: str, callback: callable):
        """Remove event callback"""
        if event_type in self.event_callbacks and callback in self.event_callbacks[event_type]:
            self.event_callbacks[event_type].remove(callback)
    
    def broadcast_database_change(self, db_event: DatabaseEvent):
        """Broadcast database change event"""
        try:
            realtime_event = RealtimeEvent(
                event_type=EventType.DATABASE_CHANGE,
                data={
                    'change_type': db_event.change_type.value,
                    'table': db_event.table,
                    'record_id': db_event.record_id,
                    'data': db_event.data,
                    'db_timestamp': db_event.timestamp.isoformat()
                }
            )
            
            # Broadcast to appropriate rooms
            if db_event.table == 'history_entries':
                self.broadcast_event(realtime_event, room='dashboard')
                self.broadcast_event(realtime_event, room='history')
            elif db_event.table == 'task_clusters':
                self.broadcast_event(realtime_event, room='tasks')
                self.broadcast_event(realtime_event, room='dashboard')
            else:
                self.broadcast_event(realtime_event, room='dashboard')
                
        except Exception as e:
            logger.error(f"Error broadcasting database change: {e}")
    
    def broadcast_task_update(self, task_data: Dict[str, Any]):
        """Broadcast current task update"""
        try:
            realtime_event = RealtimeEvent(
                event_type=EventType.TASK_UPDATE,
                data=task_data
            )
            self.broadcast_event(realtime_event, room='tasks')
            self.broadcast_event(realtime_event, room='dashboard')
        except Exception as e:
            logger.error(f"Error broadcasting task update: {e}")
    
    def broadcast_category_learned(self, category_data: Dict[str, Any]):
        """Broadcast new category learning event"""
        try:
            realtime_event = RealtimeEvent(
                event_type=EventType.CATEGORY_LEARNED,
                data=category_data
            )
            self.broadcast_event(realtime_event, room='categories')
            self.broadcast_event(realtime_event, room='dashboard')
        except Exception as e:
            logger.error(f"Error broadcasting category learned: {e}")
    
    def broadcast_semantic_search_result(self, search_results: List[Dict[str, Any]], query: str):
        """Broadcast semantic search results"""
        try:
            realtime_event = RealtimeEvent(
                event_type=EventType.SEMANTIC_SEARCH_RESULT,
                data={
                    'query': query,
                    'results': search_results,
                    'result_count': len(search_results)
                }
            )
            self.broadcast_event(realtime_event, room='search')
        except Exception as e:
            logger.error(f"Error broadcasting search results: {e}")
    
    def broadcast_analytics_update(self, analytics_data: Dict[str, Any]):
        """Broadcast analytics data update"""
        try:
            realtime_event = RealtimeEvent(
                event_type=EventType.ANALYTICS_UPDATE,
                data=analytics_data
            )
            self.broadcast_event(realtime_event, room='analytics')
            self.broadcast_event(realtime_event, room='dashboard')
        except Exception as e:
            logger.error(f"Error broadcasting analytics update: {e}")
    
    def broadcast_performance_alert(self, alert_data: Dict[str, Any]):
        """Broadcast performance alert"""
        try:
            realtime_event = RealtimeEvent(
                event_type=EventType.PERFORMANCE_ALERT,
                data=alert_data
            )
            self.broadcast_event(realtime_event, room='dashboard')
        except Exception as e:
            logger.error(f"Error broadcasting performance alert: {e}")
    
    def _get_recent_events_for_room(self, room: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent events relevant to a room"""
        try:
            # Filter events based on room relevance
            relevant_events = []
            
            for event in reversed(self.event_history[-limit*2:]):  # Look at more events to filter
                if self._is_event_relevant_to_room(event, room):
                    relevant_events.append(event.to_dict())
                    if len(relevant_events) >= limit:
                        break
            
            return relevant_events
        except Exception as e:
            logger.error(f"Error getting recent events for room {room}: {e}")
            return []
    
    def _is_event_relevant_to_room(self, event: RealtimeEvent, room: str) -> bool:
        """Check if event is relevant to a specific room"""
        try:
            if room == 'dashboard':
                return True  # Dashboard gets all events
            elif room == 'history' and event.event_type == EventType.DATABASE_CHANGE:
                return event.data.get('table') == 'history_entries'
            elif room == 'tasks' and event.event_type in [EventType.TASK_UPDATE, EventType.DATABASE_CHANGE]:
                return event.data.get('table') in ['task_clusters', 'browsing_sessions']
            elif room == 'categories' and event.event_type == EventType.CATEGORY_LEARNED:
                return True
            elif room == 'search' and event.event_type == EventType.SEMANTIC_SEARCH_RESULT:
                return True
            elif room == 'analytics' and event.event_type == EventType.ANALYTICS_UPDATE:
                return True
            
            return False
        except Exception as e:
            logger.error(f"Error checking event relevance: {e}")
            return False
    
    def _generate_overview_data(self) -> Dict[str, Any]:
        """Generate overview dashboard data"""
        try:
            return {
                'active_connections': len(self.connected_clients),
                'events_today': len([e for e in self.event_history if e.timestamp.date() == datetime.now().date()]),
                'last_update': datetime.now().isoformat(),
                'system_status': 'healthy',
                'websocket_enabled': self.socketio is not None and SOCKETIO_AVAILABLE,
                'total_events': len(self.event_history)
            }
        except Exception as e:
            logger.error(f"Error generating overview data: {e}")
            return {'error': str(e)}
    
    def _generate_analytics_data(self) -> Dict[str, Any]:
        """Generate analytics dashboard data"""
        try:
            return {
                'event_distribution': self._get_event_type_distribution(),
                'connection_stats': self.event_stats,
                'room_stats': {room: len(clients) for room, clients in self.room_subscriptions.items()}
            }
        except Exception as e:
            logger.error(f"Error generating analytics data: {e}")
            return {'error': str(e)}
    
    def _generate_category_data(self) -> Dict[str, Any]:
        """Generate category dashboard data"""
        try:
            category_events = [e for e in self.event_history if e.event_type == EventType.CATEGORY_LEARNED]
            return {
                'recent_categories': [e.data for e in category_events[-10:]],
                'total_learned': len(category_events)
            }
        except Exception as e:
            logger.error(f"Error generating category data: {e}")
            return {'error': str(e)}
    
    def _get_event_type_distribution(self) -> Dict[str, int]:
        """Get distribution of event types"""
        try:
            distribution = {}
            for event in self.event_history:
                event_type = event.event_type.value
                distribution[event_type] = distribution.get(event_type, 0) + 1
            return distribution
        except Exception as e:
            logger.error(f"Error getting event type distribution: {e}")
            return {}
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get real-time connection statistics"""
        try:
            uptime_seconds = 0
            if self.connected_clients:
                oldest_connection = min(client['connected_at'] for client in self.connected_clients.values())
                uptime_seconds = (datetime.now() - oldest_connection).total_seconds()
            
            return {
                'active_connections': len(self.connected_clients),
                'room_subscriptions': {room: len(clients) for room, clients in self.room_subscriptions.items()},
                'event_stats': self.event_stats,
                'recent_events_count': len(self.event_history),
                'uptime_seconds': uptime_seconds,
                'websocket_enabled': self.socketio is not None and SOCKETIO_AVAILABLE
            }
        except Exception as e:
            logger.error(f"Error getting connection stats: {e}")
            return {'error': str(e)}
    
    def cleanup_inactive_connections(self, timeout_minutes: int = 30):
        """Clean up inactive connections"""
        try:
            cutoff_time = datetime.now() - timedelta(minutes=timeout_minutes)
            inactive_clients = [
                client_id for client_id, client_data in self.connected_clients.items()
                if client_data['last_activity'] < cutoff_time
            ]
            
            for client_id in inactive_clients:
                # Remove from rooms
                client_subscriptions = self.connected_clients[client_id]['subscriptions']
                for room in client_subscriptions:
                    if room in self.room_subscriptions:
                        self.room_subscriptions[room].discard(client_id)
                
                del self.connected_clients[client_id]
            
            self.event_stats['connections_active'] = len(self.connected_clients)
            logger.info(f"Cleaned up {len(inactive_clients)} inactive connections")
            return len(inactive_clients)
            
        except Exception as e:
            logger.error(f"Error cleaning up inactive connections: {e}")
            return 0
    
    def get_event_history(self, event_type: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get event history, optionally filtered by type"""
        try:
            events = self.event_history
            
            if event_type:
                events = [e for e in events if e.event_type.value == event_type]
            
            # Return most recent events first
            recent_events = events[-limit:] if len(events) > limit else events
            return [event.to_dict() for event in reversed(recent_events)]
            
        except Exception as e:
            logger.error(f"Error getting event history: {e}")
            return []
    
    def clear_event_history(self):
        """Clear event history (for maintenance)"""
        try:
            old_count = len(self.event_history)
            self.event_history.clear()
            logger.info(f"Cleared {old_count} events from history")
        except Exception as e:
            logger.error(f"Error clearing event history: {e}")
    
    def get_room_info(self, room: str) -> Dict[str, Any]:
        """Get information about a specific room"""
        try:
            clients = self.room_subscriptions.get(room, set())
            client_info = []
            
            for client_id in clients:
                if client_id in self.connected_clients:
                    client_data = self.connected_clients[client_id]
                    client_info.append({
                        'client_id': client_id,
                        'connected_at': client_data['connected_at'].isoformat(),
                        'last_activity': client_data['last_activity'].isoformat(),
                        'subscriptions': list(client_data['subscriptions'])
                    })
            
            return {
                'room': room,
                'client_count': len(clients),
                'clients': client_info,
                'recent_events': self._get_recent_events_for_room(room, 5)
            }
            
        except Exception as e:
            logger.error(f"Error getting room info for {room}: {e}")
            return {'error': str(e)}
    
    def shutdown(self):
        """Shutdown the real-time service"""
        try:
            logger.info("Shutting down real-time service...")
            
            # Disconnect all clients
            for client_id in list(self.connected_clients.keys()):
                if self.socketio and SOCKETIO_AVAILABLE:
                    self.socketio.disconnect(client_id)
            
            # Clear data structures
            self.connected_clients.clear()
            self.room_subscriptions.clear()
            self.event_callbacks.clear()
            
            logger.info("Real-time service shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}") 